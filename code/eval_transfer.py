"""Cross-domain transfer probe: score a SAM3 prompt on an EXTERNAL crack dataset.

Runs the (evolved) champion prompt through the same SAM3 + both-metric pipeline
on an arbitrary image/mask dataset — e.g. CRACK500 (pavement) or an outcrop
GeoCrack subset — to show how a drill-core-evolved phrase transfers out of
domain. Reports soft tolerance-F1 + strict Dice, positives-only and all-patch.

Pass several --prompt to compare (e.g. new champion vs old champion vs a generic
phrase) on the same images in one pass.

Dataset layout (flexible): a directory of images and a directory of masks whose
files pair by stem (foo.jpg <-> foo.png, or --mask-suffix to strip/add a tag).

This covers the SAM3 transfer rows. Grounded-SAM / LISA are a different model
stack and are out of scope for this script.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_transfer.py \
        --images-dir /data/crack500/images --masks-dir /data/crack500/masks \
        --dataset crack500 \
        --prompt "dark, thin, branching crack in drill core" \
        --prompt "cracks in drill core" \
        --prompt "thin dark cracks in rock surface" \
        --out outputs/transfer_crack500.json
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
os.environ.setdefault("COREFRAC_METRIC_TOLERANCE", "2")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _index(d: Path, suffix_strip: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            stem = p.stem
            if suffix_strip and stem.endswith(suffix_strip):
                stem = stem[: -len(suffix_strip)]
            out[stem] = p.resolve()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--dataset", default="external", help="label for the report")
    ap.add_argument("--prompt", action="append", required=True, help="repeatable")
    ap.add_argument("--mask-suffix", default="", help="strip this from mask stems before pairing (e.g. _mask)")
    ap.add_argument("--mask-threshold", type=int, default=127)
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["COREFRAC_METRIC_TOLERANCE"] = str(args.tolerance)

    from problems.prompts.corefrac.utils.mask_metrics import (
        compute_mask_metrics,
    )
    from problems.prompts.corefrac.utils.sam3_tool import (
        sam3_segment_cracks,
    )

    images = _index(Path(args.images_dir), "")
    masks = _index(Path(args.masks_dir), args.mask_suffix)
    common = sorted(set(images) & set(masks))
    if args.limit:
        common = common[: args.limit]
    if not common:
        raise SystemExit("No paired image/mask stems found; check --mask-suffix.")
    print(f"[{args.dataset}] paired {len(common)} samples; prompts={len(args.prompt)}", flush=True)

    def load_mask(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            return np.asarray(im.convert("L")) > args.mask_threshold

    out: dict = {"dataset": args.dataset, "tolerance": args.tolerance, "prompts": args.prompt, "results": {}}
    for prompt in args.prompt:
        per: dict[str, dict] = {}
        t0 = time.time()
        for j, stem in enumerate(common):
            gt = load_mask(masks[stem])
            res = sam3_segment_cracks(str(images[stem]), prompt)
            pred = np.asarray(res.get("pred_mask", []), dtype=bool)
            if pred.ndim != 2 or pred.shape != gt.shape:
                pred = np.zeros(gt.shape, dtype=bool)
            m = compute_mask_metrics(pred, gt, tol=args.tolerance)
            per[stem] = {
                "is_positive": bool(gt.any()),
                "dice": m["dice"],
                "dice_strict": m["dice_strict"],
                "iou": m["iou"],
                "error": bool(res.get("error")),
            }
            if (j + 1) % 25 == 0:
                print(f"  [{prompt[:24]}] {j + 1}/{len(common)} ({time.time() - t0:.0f}s)", flush=True)
        out["results"][prompt] = per

    json.dump(out, open(args.out, "w"))
    print(f"\nwrote {args.out}")
    print(f"\n{'prompt':>40} {'dPosSoft':>9} {'dAllSoft':>9} {'dPosStr':>8} {'iouPos':>7}")
    for prompt, per in out["results"].items():
        pos = [r for r in per.values() if r["is_positive"]]
        allr = list(per.values())

        def mean(xs, k):
            v = [r[k] for r in xs]
            return sum(v) / max(1, len(v))

        print(
            f"{prompt[:40]:>40} {mean(pos, 'dice'):>9.3f} {mean(allr, 'dice'):>9.3f} "
            f"{mean(pos, 'dice_strict'):>8.3f} {mean(pos, 'iou'):>7.3f}"
        )


if __name__ == "__main__":
    main()
