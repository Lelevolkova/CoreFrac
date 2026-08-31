"""Distil the SAM3 champion prompt into a small on-prem U-Net.

Pipeline (no human labels touch the U-Net training targets):

  1. labels : run SAM3 with the champion prompt on the TRAIN (and VAL) patches
              and cache the predicted masks as pseudo-labels.
  2. train  : train a compact U-Net on (patch image -> SAM3 pseudo-mask).
              Checkpoint selection uses VAL *ground-truth* positives-only Dice
              (labels used only for model selection, never as a train target;
              switch with --select-on).
  3. eval   : score the distilled U-Net on the held-out TEST split against GT,
              with the SAME per-patch Dice/IoU metric as eval_ladder_testset.py.
  4. bench  : measure params, on-disk size, inference VRAM, latency/patch
              (GPU and CPU) for the cost / data-locality table.

Everything uses the bundle's own split manifest, dataset loader and metric, so
the distilled numbers drop straight into tab:matrix / tab:distill / tab:cost.

Typical full run on the spare 4-GPU box (one GPU is enough; SAM3 only needed
for the `labels` stage):

    GIGAEVO_PYTHON=.../corefrac-env/bin/python
    CUDA_VISIBLE_DEVICES=0 $GIGAEVO_PYTHON distill_unet.py --stage all

Stage-by-stage (e.g. if you want to inspect pseudo-labels first):

    CUDA_VISIBLE_DEVICES=0 $GIGAEVO_PYTHON distill_unet.py --stage labels
    CUDA_VISIBLE_DEVICES=0 $GIGAEVO_PYTHON distill_unet.py --stage train
    CUDA_VISIBLE_DEVICES=0 $GIGAEVO_PYTHON distill_unet.py --stage eval
    CUDA_VISIBLE_DEVICES=0 $GIGAEVO_PYTHON distill_unet.py --stage bench --bench-cpu
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

# Bundle dataset / metric wiring (matches eval_ladder_testset_soft.py: the new
# soft-metric problem package, whose compute_mask_metrics returns BOTH the soft
# tolerance-F1 ("dice") and strict pixel Dice ("dice_strict")).
PROBLEM = "corefrac"
os.environ.setdefault(
    "COREFRAC_ROOT", str(BUNDLE / "data" / "corefrac/patches")
)
os.environ.setdefault(
    "COREFRAC_SPLIT_MANIFEST",
    str(BUNDLE / "problems" / "prompts" / PROBLEM / "corefrac_split_manifest.json"),
)
os.environ.setdefault("COREFRAC_EVAL_MODE", "full")
os.environ.setdefault("COREFRAC_METRIC_TOLERANCE", "2")

CHAMPION_PROMPT = "dark, thin, branching crack in drill core"  # new soft-run champion (train soft-F1 0.5547)
NET_SIZE = 256  # square input the U-Net sees; predictions are resized back to native
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUT_DIR = BUNDLE / "outputs" / "distill"
PSEUDO_DIR = OUT_DIR / "pseudo"
CKPT_PATH = OUT_DIR / "unet_distilled.pt"
TEST_JSON = OUT_DIR / "distill_test_results.json"
COST_JSON = OUT_DIR / "cost.json"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _to_net_input(img: Image.Image) -> np.ndarray:
    """RGB PIL -> CHW float tensor at NET_SIZE x NET_SIZE, ImageNet-normalised."""
    arr = np.asarray(img.resize((NET_SIZE, NET_SIZE), Image.Resampling.BILINEAR))
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).copy()


def _mask_to_net(mask: np.ndarray) -> np.ndarray:
    """Bool mask (native) -> float {0,1} at NET_SIZE x NET_SIZE (nearest)."""
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (NET_SIZE, NET_SIZE), Image.Resampling.NEAREST
    )
    return (np.asarray(im) > 127).astype(np.float32)


def _resize_pred_to_native(prob_small: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Probability map at NET_SIZE -> bool mask at native (H, W) via 0.5 threshold."""
    height, width = shape
    hard = (prob_small > 0.5).astype(np.uint8) * 255
    im = Image.fromarray(hard).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(im) > 127


def _pseudo_path(split: str, sample_id: str) -> Path:
    return PSEUDO_DIR / split / f"{sample_id}.png"


# --------------------------------------------------------------------------- #
# Stage 1: SAM3 pseudo-labels
# --------------------------------------------------------------------------- #
def stage_labels(splits: list[str], prompt: str) -> None:
    os.environ.setdefault("SAM3_DEVICE", "cuda")
    os.environ.setdefault("SAM3_MODEL_NAME", "facebook/sam3")
    os.environ.setdefault("SAM3_CONFIDENCE", "0.5")
    os.environ.setdefault("SAM3_CUDA_INDEX", "0")
    os.environ.setdefault("SAM3_GPU_SLOT_DIR", "/tmp/corefrac_distill_slots")

    from problems.prompts.corefrac.utils.dataset import load_split
    from problems.prompts.corefrac.utils.sam3_tool import (
        sam3_segment_cracks,
    )

    root = Path(os.environ["COREFRAC_ROOT"])
    for split in splits:
        samples = sorted(load_split(root, split), key=lambda s: s.sample_id)
        out_dir = PSEUDO_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_done = 0
        for i, s in enumerate(samples):
            out = _pseudo_path(split, s.sample_id)
            if out.exists():
                continue
            res = sam3_segment_cracks(str(s.image_path), prompt)
            pred = np.asarray(res.get("pred_mask", []), dtype=bool)
            with Image.open(s.image_path) as im:
                shape = (im.height, im.width)
            if pred.ndim != 2 or pred.shape != shape:
                pred = np.zeros(shape, dtype=bool)
            Image.fromarray(pred.astype(np.uint8) * 255).save(out)
            n_done += 1
            if (i + 1) % 20 == 0:
                print(f"[labels:{split}] {i + 1}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[labels:{split}] wrote {n_done} new / {len(samples)} total -> {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Compact U-Net (pure torch, no external segmentation libs)
# --------------------------------------------------------------------------- #
def _build_unet(width: int, depth: int):
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, width=32, depth=4, in_ch=3):
            super().__init__()
            import torch

            self._torch = torch
            chs = [width * (2**i) for i in range(depth + 1)]
            self.downs = nn.ModuleList()
            prev = in_ch
            for c in chs[:-1]:
                self.downs.append(DoubleConv(prev, c))
                prev = c
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(chs[-2], chs[-1])
            rev = chs[::-1]
            self.upconvs = nn.ModuleList()
            self.ups = nn.ModuleList()
            for i in range(depth):
                self.upconvs.append(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2))
                self.ups.append(DoubleConv(rev[i], rev[i + 1]))
            self.head = nn.Conv2d(chs[0], 1, 1)

        def forward(self, x):
            torch = self._torch
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for i, (upc, up) in enumerate(zip(self.upconvs, self.ups)):
                x = upc(x)
                skip = skips[-(i + 1)]
                if x.shape[-2:] != skip.shape[-2:]:
                    x = torch.nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="nearest"
                    )
                x = up(torch.cat([skip, x], dim=1))
            return self.head(x)

    return UNet(width=width, depth=depth)


def _dice_loss(logits, target, eps=1.0):
    import torch

    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


# --------------------------------------------------------------------------- #
# Stage 2: train
# --------------------------------------------------------------------------- #
def _load_split_arrays(split: str, use_pseudo: bool):
    """Return list of (sample_id, net_input CHW, target HxW@NET, gt_bool_native, native_shape)."""
    from problems.prompts.corefrac.utils.dataset import (
        load_ground_truth_mask,
        load_split,
    )

    root = Path(os.environ["COREFRAC_ROOT"])
    samples = sorted(load_split(root, split), key=lambda s: s.sample_id)
    items = []
    for s in samples:
        img = _load_rgb(s.image_path)
        native = (img.height, img.width)
        x = _to_net_input(img)
        gt_native = load_ground_truth_mask(s.mask_path)
        if use_pseudo:
            pp = _pseudo_path(split, s.sample_id)
            if not pp.exists():
                raise FileNotFoundError(
                    f"Missing pseudo-label {pp}. Run `--stage labels` first."
                )
            with Image.open(pp) as pm:
                tgt_native = np.asarray(pm.convert("L")) > 127
            target = _mask_to_net(tgt_native)
        else:
            target = _mask_to_net(gt_native)
        items.append((s.sample_id, x, target, gt_native, native))
    return items


def _pos_only_dice(net, items, device, batch=8):
    """Mean positives-only soft tolerance-F1 + success-rate on native-res GT."""
    import torch

    from problems.prompts.corefrac.utils.mask_metrics import (
        compute_mask_metrics,
    )

    net.eval()
    pos_d, hits, n_pos = [], 0, 0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            xb = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
            prob = torch.sigmoid(net(xb)).cpu().numpy()[:, 0]
            for j, c in enumerate(chunk):
                _sid, _x, _t, gt_native, native = c
                pred = _resize_pred_to_native(prob[j], native)
                m = compute_mask_metrics(pred, gt_native)
                if gt_native.any():
                    n_pos += 1
                    pos_d.append(m["dice"])
                    if m["dice"] >= 0.5:
                        hits += 1
    md = float(np.mean(pos_d)) if pos_d else 0.0
    sr = 100.0 * hits / n_pos if n_pos else 0.0
    return md, sr


def stage_train(args) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    use_pseudo = args.select_on != "noop"  # always train on pseudo
    train_items = _load_split_arrays("train", use_pseudo=True)
    val_items = _load_split_arrays("val", use_pseudo=False)

    pos_frac = float(np.mean([t[2].mean() for t in train_items]))
    pos_weight = min(max((1.0 - pos_frac) / max(pos_frac, 1e-4), 1.0), 30.0)
    print(
        f"train={len(train_items)} val={len(val_items)} pos_frac={pos_frac:.4f} "
        f"pos_weight={pos_weight:.1f} device={device}",
        flush=True,
    )

    net = _build_unet(args.width, args.depth).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"U-Net width={args.width} depth={args.depth} params={n_params / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    X = torch.from_numpy(np.stack([t[1] for t in train_items]))
    Y = torch.from_numpy(np.stack([t[2] for t in train_items]))[:, None]
    n = X.shape[0]

    best_sel, best_state = -1.0, None
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            # cheap flip augmentation
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
                yb = torch.flip(yb, dims=[3])
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[2])
                yb = torch.flip(yb, dims=[2])
            logits = net(xb)
            loss = bce(logits, yb) + _dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(idx)
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            val_d, val_sr = _pos_only_dice(net, val_items, device)
            if args.select_on == "valgt":
                sel = val_d
            else:  # trainloss
                sel = -ep_loss / n
            tag = ""
            if sel > best_sel:
                best_sel = sel
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                tag = " *best"
            print(
                f"ep {ep + 1:>3}/{args.epochs} loss={ep_loss / n:.4f} "
                f"val_posDice={val_d:.3f} val_SR={val_sr:.1f}%{tag}",
                flush=True,
            )

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save(
        {
            "state_dict": best_state,
            "width": args.width,
            "depth": args.depth,
            "net_size": NET_SIZE,
            "prompt": args.prompt,
            "n_params": n_params,
            "select_on": args.select_on,
            "best_sel": best_sel,
        },
        CKPT_PATH,
    )
    print(f"saved -> {CKPT_PATH}  (best_sel={best_sel:.3f})", flush=True)


# --------------------------------------------------------------------------- #
# Stage 3: held-out test eval (same metric/shape as eval_ladder_testset.py)
# --------------------------------------------------------------------------- #
def _load_net_from_ckpt(device):
    import torch

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    net = _build_unet(ckpt["width"], ckpt["depth"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt


def stage_eval(args) -> None:
    import torch

    from problems.prompts.corefrac.utils.mask_metrics import (
        compute_mask_metrics,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ckpt = _load_net_from_ckpt(device)
    test_items = _load_split_arrays("test", use_pseudo=False)

    results: dict[str, dict] = {}
    soft_all, soft_pos, strict_all, strict_pos, pos_iou = [], [], [], [], []
    n_pos = n_emp = pos_hit = empty_fp = 0
    with torch.no_grad():
        for i in range(0, len(test_items), args.batch):
            chunk = test_items[i : i + args.batch]
            xb = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
            prob = torch.sigmoid(net(xb)).cpu().numpy()[:, 0]
            for j, c in enumerate(chunk):
                sid, _x, _t, gt_native, native = c
                pred = _resize_pred_to_native(prob[j], native)
                m = compute_mask_metrics(pred, gt_native)
                is_pos = bool(gt_native.any())
                results[sid] = {
                    "is_positive": is_pos,
                    "dice": m["dice"],              # soft tolerance-F1 (primary)
                    "dice_strict": m["dice_strict"],
                    "iou": m["iou"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                }
                soft_all.append(m["dice"])
                strict_all.append(m["dice_strict"])
                if is_pos:
                    n_pos += 1
                    soft_pos.append(m["dice"])
                    strict_pos.append(m["dice_strict"])
                    pos_iou.append(m["iou"])
                    if m["dice"] >= 0.5:
                        pos_hit += 1
                else:
                    n_emp += 1
                    if m["dice"] < 0.5:
                        empty_fp += 1

    def _mean(x):
        return float(np.mean(x)) if x else 0.0

    summary = {
        "model": "distilled_unet",
        "prompt": ckpt.get("prompt"),
        "n_params": ckpt.get("n_params"),
        "n_test": len(test_items),
        "n_pos": n_pos,
        "n_empty": n_emp,
        "tolerance_px": int(os.environ.get("COREFRAC_METRIC_TOLERANCE", "2")),
        "soft_dice_all": _mean(soft_all),
        "soft_dice_pos": _mean(soft_pos),
        "strict_dice_all": _mean(strict_all),
        "strict_dice_pos": _mean(strict_pos),
        "iou_pos": _mean(pos_iou),
        "success_rate_pct": 100.0 * pos_hit / n_pos if n_pos else 0.0,
        "empty_fp": empty_fp,
    }
    TEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(TEST_JSON, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote -> {TEST_JSON}", flush=True)
    print(
        "\nFor tab:matrix / tab:distill (distilled U-Net, held-out test):\n"
        f"  positives-only soft-F1 = {summary['soft_dice_pos']:.3f}  (all-patch {summary['soft_dice_all']:.3f})\n"
        f"  positives-only strict Dice = {summary['strict_dice_pos']:.3f}  (all-patch {summary['strict_dice_all']:.3f})\n"
        f"  positives IoU = {summary['iou_pos']:.3f}\n"
        f"  success rate (soft posF1>=0.5) = {summary['success_rate_pct']:.1f}%",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Stage 4: cost / latency / VRAM
# --------------------------------------------------------------------------- #
def _bench_device(net, device, n_iter=100, warmup=10):
    import torch

    net = net.to(device).eval()
    x = torch.randn(1, 3, NET_SIZE, NET_SIZE, device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            net(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    out = {
        "device": device,
        "latency_ms_median": float(np.median(ts)),
        "latency_ms_p90": float(np.percentile(ts, 90)),
    }
    if device.startswith("cuda"):
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return out


def stage_bench(args) -> None:
    import torch

    net, ckpt = _load_net_from_ckpt("cpu")
    n_params = ckpt.get("n_params", sum(p.numel() for p in net.parameters()))
    size_mb = CKPT_PATH.stat().st_size / 1e6

    cost = {
        "model": "distilled_unet",
        "n_params_M": n_params / 1e6,
        "ckpt_size_mb": size_mb,
        "net_input": NET_SIZE,
        "benches": [],
    }
    if torch.cuda.is_available():
        cost["benches"].append(_bench_device(_build_unet(ckpt["width"], ckpt["depth"]).eval(), "cuda:0"))
        cost["gpu_name"] = torch.cuda.get_device_name(0)
    if args.bench_cpu:
        torch.set_num_threads(args.cpu_threads)
        cost["benches"].append(_bench_device(net, "cpu"))
        cost["cpu_threads"] = args.cpu_threads

    COST_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cost, open(COST_JSON, "w"), indent=2)
    print(json.dumps(cost, indent=2), flush=True)
    print(f"\nwrote -> {COST_JSON}", flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["labels", "train", "eval", "bench", "all"], default="all")
    ap.add_argument("--prompt", default=CHAMPION_PROMPT, help="champion prompt for pseudo-labels")
    ap.add_argument("--width", type=int, default=32, help="U-Net base channel width")
    ap.add_argument("--depth", type=int, default=4, help="U-Net encoder depth")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--select-on", choices=["valgt", "trainloss"], default="valgt")
    ap.add_argument("--bench-cpu", action="store_true", help="also measure CPU latency")
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()

    if args.stage in ("labels", "all"):
        stage_labels(["train", "val"], args.prompt)
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)
    if args.stage in ("bench", "all"):
        stage_bench(args)


if __name__ == "__main__":
    main()
