"""Merge eval_ladder_testset_soft.py shards and report soft + strict metrics.

Reads one or more shard JSONs (default: /tmp/eval_soft_shard_*.json) and prints,
per ladder level, the held-out test numbers for the paper:

  dPosSoft / dAllSoft  - tolerance-F1 (primary)
  dPosStrict / dAllStrict - strict pixel Dice (reference)
  iouPos               - strict IoU on positives
  SR%                  - success rate (soft dPos >= 0.5)
  emptyFP              - empty patches mis-segmented

Also writes a merged test_eval_results_soft.json (same schema as the shards but
with all results combined) so downstream tooling has one file.

Usage:
    python aggregate_test_soft.py --shards /tmp/eval_soft_shard_*.json \
        --out test_eval_results_soft.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None, help="shard JSON paths (glob ok)")
    ap.add_argument("--out", default=str(BUNDLE / "test_eval_results_soft.json"))
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.shards or ["/tmp/eval_soft_shard_*.json"]:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit("No shard files matched; pass --shards.")

    merged_res: dict[str, dict] = {}
    ladder = None
    tol = None
    n_total = None
    for p in paths:
        d = json.load(open(p))
        ladder = ladder or d["ladder"]
        tol = tol or d.get("tolerance")
        n_total = n_total or d.get("n_total_test")
        merged_res.update(d["results"])

    labels = [lvl["label"] for lvl in ladder]
    prompts = {lvl["label"]: lvl["prompt"] for lvl in ladder}
    trainfit = {lvl["label"]: lvl.get("train_fitness") for lvl in ladder}

    n = len(merged_res)
    n_pos = sum(1 for r in merged_res.values() if r["is_positive"])
    n_emp = n - n_pos
    print(f"patches: total={n} positive={n_pos} empty={n_emp}  tol={tol}px  shards={len(paths)}")
    if n_total and n != n_total:
        print(f"  WARNING: merged {n} != expected {n_total} (missing shards?)")
    print()
    hdr = (
        f"{'level':>14} {'train':>7} {'dPosSoft':>9} {'dAllSoft':>9} "
        f"{'dPosStr':>8} {'dAllStr':>8} {'iouPos':>7} {'SR%':>6} {'emptyFP':>8}"
    )
    print(hdr)
    summary = {}
    for lab in labels:
        soft_pos, soft_all, str_pos, str_all, iou_pos = [], [], [], [], []
        empty_fp = pos_hit = 0
        for r in merged_res.values():
            lv = r["levels"][lab]
            soft_all.append(lv["dice"])
            str_all.append(lv["dice_strict"])
            if r["is_positive"]:
                soft_pos.append(lv["dice"])
                str_pos.append(lv["dice_strict"])
                iou_pos.append(lv["iou"])
                if lv["dice"] >= 0.5:
                    pos_hit += 1
            elif lv["dice"] < 0.5:
                empty_fp += 1

        def mean(x):
            return sum(x) / len(x) if x else 0.0

        row = {
            "train_fitness": trainfit[lab],
            "dPosSoft": mean(soft_pos),
            "dAllSoft": mean(soft_all),
            "dPosStrict": mean(str_pos),
            "dAllStrict": mean(str_all),
            "iouPos": mean(iou_pos),
            "success_rate_pct": 100.0 * pos_hit / n_pos if n_pos else 0.0,
            "empty_fp": empty_fp,
        }
        summary[lab] = row
        tf = f"{trainfit[lab]:.3f}" if trainfit[lab] is not None else "  -  "
        print(
            f"{lab:>14} {tf:>7} {row['dPosSoft']:>9.3f} {row['dAllSoft']:>9.3f} "
            f"{row['dPosStrict']:>8.3f} {row['dAllStrict']:>8.3f} {row['iouPos']:>7.3f} "
            f"{row['success_rate_pct']:>6.1f} {empty_fp:>4}/{n_emp:<3}"
        )
    print()
    for lab in labels:
        print(f'{lab}: "{prompts[lab]}"')

    json.dump(
        {"tolerance": tol, "n_total_test": n, "n_pos": n_pos, "n_empty": n_emp,
         "ladder": ladder, "summary": summary, "results": merged_res},
        open(args.out, "w"), indent=2, ensure_ascii=False,
    )
    print(f"\nmerged -> {args.out}")


if __name__ == "__main__":
    main()
