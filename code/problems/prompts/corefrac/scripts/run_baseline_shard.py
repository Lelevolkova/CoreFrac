"""Run CoreFrac SAM3 prompt baselines on one GPU shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("COREFRAC_ROOT", str(REPO_ROOT / "data" / "corefrac" / "patches"))
os.environ.setdefault("SAM3_DEVICE", "cuda")

from problems.prompts.corefrac import (  # noqa: E402
    config,
)
from problems.prompts.corefrac.utils.dataset import (  # noqa: E402
    build_patch_samples,
    load_ground_truth_mask,
    load_split,
)
from problems.prompts.corefrac.utils.mask_metrics import (
    compute_mask_metrics,  # noqa: E402
)
from problems.prompts.corefrac.utils.sam3_tool import (
    sam3_segment_cracks,  # noqa: E402
)

ZERO_METRICS = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}


def _coerce_pred_mask(
    tool_output: dict[str, Any], shape: tuple[int, int]
) -> np.ndarray:
    pred_mask = np.asarray(tool_output.get("pred_mask", []), dtype=bool)
    if pred_mask.ndim != 2 or pred_mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    return pred_mask


def load_samples_for_splits(splits: list[str]):
    samples = []
    for split in splits:
        for sample in load_split(split=split):
            samples.append((split, sample))
    return samples


def eval_full(prompt_name: str, prompt: str, split: str, sample) -> dict[str, Any]:
    gt_mask = load_ground_truth_mask(sample.mask_path)
    tool_output = sam3_segment_cracks(str(sample.image_path), prompt)
    pred_mask = _coerce_pred_mask(tool_output, gt_mask.shape)
    metrics = (
        compute_mask_metrics(pred_mask, gt_mask)
        if tool_output.get("success")
        else ZERO_METRICS
    )
    return {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "split": split,
        "sample_id": sample.sample_id,
        **metrics,
        "success": bool(tool_output.get("success", False)),
        "n_instances": int(tool_output.get("n_instances", 0)),
        "latency_sec": float(tool_output.get("latency_sec", 0.0)),
        "error": tool_output.get("error"),
    }


def eval_patch_stitch(
    prompt_name: str, prompt: str, split: str, sample
) -> dict[str, Any]:
    gt_mask = load_ground_truth_mask(sample.mask_path)
    stitched = np.zeros_like(gt_mask, dtype=bool)
    any_success = False
    latency = 0.0
    errors = []
    n_instances = 0
    for patch in build_patch_samples([sample], materialize=True):
        tool_output = sam3_segment_cracks(str(patch.image_path), prompt)
        patch_shape = (patch.y1 - patch.y0, patch.x1 - patch.x0)
        pred_patch = _coerce_pred_mask(tool_output, patch_shape)
        stitched[patch.y0 : patch.y1, patch.x0 : patch.x1] |= pred_patch
        any_success = any_success or bool(tool_output.get("success", False))
        latency += float(tool_output.get("latency_sec", 0.0))
        n_instances += int(tool_output.get("n_instances", 0))
        if tool_output.get("error"):
            errors.append(str(tool_output["error"]))
    metrics = compute_mask_metrics(stitched, gt_mask) if any_success else ZERO_METRICS
    return {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "split": split,
        "sample_id": sample.sample_id,
        **metrics,
        "success": any_success,
        "n_instances": n_instances,
        "latency_sec": latency,
        "error": "; ".join(errors) if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--eval-mode", choices=["full", "patch", "patch_stitch"], default="full"
    )
    parser.add_argument(
        "--prompts", default="all", help="Comma-separated baseline names or 'all'"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", default=None)
    args = parser.parse_args()

    if os.environ.get("SAM3_DEVICE", "cuda") == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only run requested but CUDA is unavailable")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    prompt_names = (
        list(config.PROMPT_BASELINES)
        if args.prompts == "all"
        else [p.strip() for p in args.prompts.split(",") if p.strip()]
    )
    prompts = [(name, config.PROMPT_BASELINES[name]) for name in prompt_names]

    all_jobs = []
    for split, sample in load_samples_for_splits(splits):
        for prompt_name, prompt in prompts:
            all_jobs.append((prompt_name, prompt, split, sample))
    selected = [
        item
        for idx, item in enumerate(all_jobs)
        if idx % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = (
        Path(args.progress) if args.progress else out.with_suffix(".progress.jsonl")
    )
    progress.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    with out.open("w", encoding="utf-8") as fh:
        with progress.open("w", encoding="utf-8") as pf:
            for idx, (prompt_name, prompt, split, sample) in enumerate(
                selected, start=1
            ):
                if args.eval_mode == "full":
                    row = eval_full(prompt_name, prompt, split, sample)
                else:
                    row = eval_patch_stitch(prompt_name, prompt, split, sample)
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                status = {
                    "shard_index": args.shard_index,
                    "done": idx,
                    "total": len(selected),
                    "elapsed_sec": time.time() - started,
                    "last_prompt": prompt_name,
                    "last_sample_id": sample.sample_id,
                    "last_success": row["success"],
                    "last_dice": row["dice"],
                }
                pf.write(json.dumps(status) + "\n")
                pf.flush()
                print(json.dumps(status), flush=True)

    summary = {
        "n": len(rows),
        "shard_index": args.shard_index,
        "eval_mode": args.eval_mode,
        "n_errors": sum(1 for r in rows if r.get("error")),
        "n_success": sum(1 for r in rows if r.get("success")),
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
