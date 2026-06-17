"""Open-weight zero-shot segmentation baselines on the CoreFrac 176-patch test.

On-prem, open-weight alternatives to the VLM "draw-the-mask" row, all runnable from
transformers 4.38 (no SAM 3 needed):

  clipseg    : CLIPSeg (text -> dense mask). Decision threshold calibrated on val.
  owlv2_sam  : OWLv2 open-vocab detection (text -> boxes) -> SAM (boxes -> masks),
               unioned. Grounded-SAM analog. Fixed box score threshold.
  florence2  : Florence-2 <REFERRING_EXPRESSION_SEGMENTATION> (text -> polygons),
               rasterized. Deterministic (beam search), no threshold.

Same test split, native-resolution scoring, soft tolerance-F1 (tau=2) + strict pixel
Dice as the paper's other rows, so the numbers drop straight into tab:matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
PROBLEM = "corefrac"
os.environ.setdefault("COREFRAC_ROOT", str(BUNDLE / "data" / "corefrac/patches"))
os.environ.setdefault(
    "COREFRAC_SPLIT_MANIFEST",
    str(BUNDLE / "problems" / "prompts" / PROBLEM / "corefrac_split_manifest.json"),
)
os.environ.setdefault("COREFRAC_METRIC_TOLERANCE", "2")


def _load_split_items(split: str):
    from problems.prompts.corefrac.utils.dataset import (
        load_ground_truth_mask,
        load_split,
    )
    root = Path(os.environ["COREFRAC_ROOT"])
    out = []
    for s in sorted(load_split(root, split), key=lambda s: s.sample_id):
        with Image.open(s.image_path) as im:
            rgb = im.convert("RGB")
        gt = load_ground_truth_mask(s.mask_path)
        out.append((s.sample_id, rgb, gt))
    return out


def _metrics(pred, gt):
    from problems.prompts.corefrac.utils.mask_metrics import (
        compute_mask_metrics,
    )
    return compute_mask_metrics(pred, gt)


def _summary(results: dict) -> dict:
    pos_soft, pos_str, pos_iou, all_soft = [], [], [], []
    n_pos = n_emp = hit = empty_fp = 0
    for r in results.values():
        all_soft.append(r["dice"])
        if r["is_positive"]:
            n_pos += 1
            pos_soft.append(r["dice"]); pos_str.append(r["dice_strict"]); pos_iou.append(r["iou"])
            if r["dice"] >= 0.5:
                hit += 1
        else:
            n_emp += 1
            if r["dice"] < 0.5:
                empty_fp += 1
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "soft_pos": round(m(pos_soft), 4), "soft_all": round(m(all_soft), 4),
        "strict_pos": round(m(pos_str), 4), "iou_pos": round(m(pos_iou), 4),
        "success_rate_pct": round(100 * hit / n_pos, 1) if n_pos else 0.0,
        "empty_fp": empty_fp, "n_pos": n_pos, "n_empty": n_emp,
    }


# --------------------------------------------------------------------------- #
def build_clipseg(prompt, device):
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()

    def prob_map(pil, hw):
        inp = proc(text=[prompt], images=[pil], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inp).logits
        p = torch.sigmoid(logits).squeeze().float().cpu().numpy()
        h, w = hw
        im = Image.fromarray((p * 255).astype("uint8")).resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255.0
    return prob_map  # caller thresholds (val-calibrated)


def build_owlv2_sam(prompt, device, box_thr=0.1):
    import torch
    from transformers import (
        Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor,
    )
    owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    def mask_fn(pil, hw):
        h, w = hw
        inp = owl_proc(text=[[prompt]], images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = owl(**inp)
        res = owl_proc.post_process_object_detection(
            out, target_sizes=torch.tensor([[h, w]]).to(device), threshold=box_thr)[0]
        boxes = res["boxes"].detach().cpu().tolist()
        if not boxes:
            return np.zeros((h, w), dtype=bool)
        sinp = sam_proc(pil, input_boxes=[boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            sout = sam(**sinp)
        masks = sam_proc.image_processor.post_process_masks(
            sout.pred_masks.cpu(), sinp["original_sizes"].cpu(), sinp["reshaped_input_sizes"].cpu())[0]
        m = masks.numpy()
        m = m.reshape(-1, m.shape[-2], m.shape[-1]).any(axis=0)
        if m.shape != (h, w):
            im = Image.fromarray((m * 255).astype("uint8")).resize((w, h), Image.Resampling.NEAREST)
            m = np.asarray(im) > 127
        return m
    return mask_fn


def build_florence2(prompt, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Florence-2 remote code declares a hard `flash_attn` import that check_imports
    # rejects on boxes without it; at runtime with eager attention it is never used.
    import importlib
    from unittest.mock import patch
    import transformers.dynamic_module_utils as _dmu
    from transformers.dynamic_module_utils import get_imports

    # Non-interactive runs (nohup) have no TTY: the "run custom code? [y/N]" prompt
    # reads EOF -> "N" -> ValueError (and arms a SIGALRM that kills generate()).
    # resolve_trust_remote_code is imported BY NAME into the auto modules, so patch
    # every binding site to force-trust.
    def _always_true(*a, **k):
        return True
    _dmu.resolve_trust_remote_code = _always_true
    _dmu._raise_timeout_error = lambda *a, **k: None  # neutralize CI/no-TTY alarm path
    for _mod in ("transformers.models.auto.auto_factory",
                 "transformers.models.auto.tokenization_auto",
                 "transformers.models.auto.image_processing_auto",
                 "transformers.models.auto.processing_auto",
                 "transformers.models.auto.feature_extraction_auto",
                 "transformers.models.auto.configuration_auto"):
        try:
            _m = importlib.import_module(_mod)
            if hasattr(_m, "resolve_trust_remote_code"):
                _m.resolve_trust_remote_code = _always_true
        except Exception:
            pass

    def _no_flash(filename):
        imports = get_imports(filename)
        return [i for i in imports if i != "flash_attn"]

    dtype = torch.float16 if device == "cuda" else torch.float32
    with patch("transformers.dynamic_module_utils.get_imports", _no_flash):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager").to(device).eval()
        proc = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"

    def mask_fn(pil, hw):
        h, w = hw
        inp = proc(text=task + prompt, images=pil, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            gen = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                 max_new_tokens=1024, num_beams=3, do_sample=False)
        txt = proc.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(txt, task=task, image_size=(pil.width, pil.height))
        polys = parsed.get(task, {}).get("polygons", [])
        canvas = Image.new("L", (pil.width, pil.height), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        for inst in polys:
            for poly in inst:
                pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    d.polygon(pts, fill=255)
                elif len(pts) == 2:
                    d.line(pts, fill=255, width=3)
        m = np.asarray(canvas) > 127
        if m.shape != (h, w):
            im = Image.fromarray((m * 255).astype("uint8")).resize((w, h), Image.Resampling.NEAREST)
            m = np.asarray(im) > 127
        return m
    return mask_fn


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clipseg", "owlv2_sam", "florence2"])
    ap.add_argument("--prompt", default="dark, thin, branching crack in drill core")
    ap.add_argument("--box-thr", type=float, default=0.1, help="owlv2 box score threshold")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = _load_split_items("test")
    if args.limit:
        test = test[: args.limit]
    print(f"[{args.model}] prompt={args.prompt!r} test={len(test)} device={device}", flush=True)

    results: dict = {}
    chosen_thr = None
    t0 = time.time()

    if args.model == "clipseg":
        prob_map = build_clipseg(args.prompt, device)
        val = _load_split_items("val")
        if args.limit:
            val = val[: args.limit]
        val_probs = [(prob_map(pil, gt.shape), gt) for _sid, pil, gt in val]
        best_thr, best_f1 = 0.5, -1.0
        for thr in [round(t, 2) for t in np.arange(0.10, 0.71, 0.05)]:
            soft = [_metrics(p > thr, gt)["dice"] for p, gt in val_probs]
            f1 = float(np.mean(soft))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        chosen_thr = best_thr
        print(f"[clipseg] val-calibrated threshold={best_thr} (val all-soft={best_f1:.3f})", flush=True)
        for sid, pil, gt in test:
            pred = prob_map(pil, gt.shape) > best_thr
            m = _metrics(pred, gt)
            results[sid] = {"is_positive": bool(gt.any()), "dice": m["dice"],
                            "dice_strict": m["dice_strict"], "iou": m["iou"]}
    else:
        mask_fn = (build_owlv2_sam(args.prompt, device, args.box_thr) if args.model == "owlv2_sam"
                   else build_florence2(args.prompt, device))
        chosen_thr = args.box_thr if args.model == "owlv2_sam" else None
        for i, (sid, pil, gt) in enumerate(test):
            pred = mask_fn(pil, gt.shape)
            m = _metrics(pred, gt)
            results[sid] = {"is_positive": bool(gt.any()), "dice": m["dice"],
                            "dice_strict": m["dice_strict"], "iou": m["iou"]}
            if (i + 1) % 20 == 0:
                print(f"[{args.model}] {i + 1}/{len(test)} in {time.time() - t0:.0f}s", flush=True)

    summary = _summary(results)
    summary.update(model=args.model, prompt=args.prompt, threshold=chosen_thr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[{args.model}] DONE in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
