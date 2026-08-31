"""Held-out test-set eval of a prompt ladder under the SOFT (tolerance) metric.

Successor to eval_ladder_testset.py for the corefrac run. For every patch of
the test split named by COREFRAC_SPLIT_MANIFEST (172 patches under the
source-grouped split) it runs SAM3 once per ladder prompt and records BOTH:

  * soft tolerance-F1 (the new primary "dice"; tol = COREFRAC_METRIC_TOLERANCE px)
  * strict pixel Dice ("dice_strict") and strict IoU ("iou")

so the paper table can lead with the soft metric and report strict alongside.

Ladder source: --ladder JSON (default frontier_ladder_soft.json, produced by
extract_frontier.py). Add --extra-prompts to score arbitrary baseline prompts on
the same split (e.g. the OLD strict-run champion, for a like-for-like bridge).

Run one shard per GPU and merge with aggregate_test_soft.py:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python eval_ladder_testset_soft.py \
        --shard $i --nshards 4 --out /tmp/eval_soft_shard_$i.json &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
PROBLEM = "corefrac"


def _load_ladder(path: Path, extra_prompts: list[str]) -> list[dict]:
    steps: list[dict] = []
    if path and path.is_file():
        for s in json.load(open(path)):
            steps.append({"label": s["label"], "prompt": s["prompt"], "train_fitness": s.get("train_fitness")})
    for p in extra_prompts:
        steps.append({"label": f"extra:{p[:24]}", "prompt": p, "train_fitness": None})
    if not steps:
        raise SystemExit("Empty ladder: pass --ladder and/or --extra-prompts")
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ladder", default=str(BUNDLE / "frontier_ladder_soft.json"))
    ap.add_argument("--extra-prompts", nargs="*", default=[])
    ap.add_argument("--tolerance", type=int, default=2, help="soft-metric slack in px")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(BUNDLE))
    os.environ.setdefault(
        "COREFRAC_ROOT", str(BUNDLE / "data" / "corefrac/patches")
    )
    # setdefault so callers (e.g. the B7 maintenance runner) can point this at a
    # domain-specific split manifest; otherwise the test split of the manifest.
    os.environ.setdefault(
        "COREFRAC_SPLIT_MANIFEST",
        str(BUNDLE / "problems" / "prompts" / PROBLEM / "corefrac_split_manifest.json"),
    )
    os.environ["COREFRAC_EVAL_MODE"] = "full"
    os.environ["COREFRAC_METRIC_TOLERANCE"] = str(args.tolerance)
    os.environ.setdefault("SAM3_DEVICE", "cuda")
    os.environ.setdefault("SAM3_MODEL_NAME", "facebook/sam3")
    os.environ.setdefault("SAM3_CONFIDENCE", "0.5")
    os.environ.setdefault("SAM3_CUDA_INDEX", "0")
    os.environ.setdefault("SAM3_GPU_SLOT_DIR", f"/tmp/corefrac_eval_soft_slots_s{args.shard}")

    import numpy as np

    from problems.prompts.corefrac.utils.dataset import (
        load_ground_truth_mask,
        load_split,
    )
    from problems.prompts.corefrac.utils.mask_metrics import (
        compute_mask_metrics,
    )
    from problems.prompts.corefrac.utils.sam3_tool import (
        sam3_segment_cracks,
    )

    ladder = _load_ladder(Path(args.ladder), args.extra_prompts)
    root = Path(os.environ["COREFRAC_ROOT"])
    test = sorted(load_split(root, "test"), key=lambda s: s.sample_id)
    shard = [s for i, s in enumerate(test) if i % args.nshards == args.shard]

    def predict(image_path, prompt, shape):
        out = sam3_segment_cracks(str(image_path), prompt)
        pred = np.asarray(out.get("pred_mask", []), dtype=bool)
        if pred.ndim != 2 or pred.shape != shape:
            pred = np.zeros(shape, dtype=bool)
        return pred, bool(out.get("error"))

    res: dict[str, dict] = {}
    t0 = time.time()
    for j, s in enumerate(shard):
        gt = load_ground_truth_mask(s.mask_path)
        rec = {"is_positive": bool(gt.any()), "levels": {}}
        for lvl in ladder:
            pred, err = predict(s.image_path, lvl["prompt"], gt.shape)
            m = compute_mask_metrics(pred, gt, tol=args.tolerance)
            rec["levels"][lvl["label"]] = {
                "dice": m["dice"],              # soft tolerance-F1 (primary)
                "dice_strict": m["dice_strict"],
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "error": err,
            }
        res[s.sample_id] = rec
        if (j + 1) % 10 == 0:
            print(f"[shard {args.shard}] {j + 1}/{len(shard)} in {time.time() - t0:.0f}s", flush=True)

    json.dump(
        {
            "shard": args.shard,
            "n_total_test": len(test),
            "tolerance": args.tolerance,
            "ladder": ladder,
            "results": res,
        },
        open(args.out, "w"),
    )
    print(f"[shard {args.shard}] DONE {len(shard)}/{len(test)} in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
