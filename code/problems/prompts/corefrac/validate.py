"""Evaluate evolved SAM3 prompts on geological core fracture samples."""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean

import numpy as np

from problems.prompts.corefrac import (
    config as problem_config,
)
from problems.prompts.corefrac.utils.dataset import (
    CrackSample,
    build_patch_samples,
    load_ground_truth_mask,
    load_split,
)
from problems.prompts.corefrac.utils.mask_metrics import (
    aggregate_metrics,
    compute_mask_metrics,
)
from problems.prompts.corefrac.utils.prompt_validation import (
    validate_sam3_prompt,
)
from problems.prompts.corefrac.utils.sam3_tool import (
    sam3_segment_cracks,
)


def load_train_samples(
    n_samples: int | None = None,
    *,
    dataset_root: Path | None = None,
) -> list[CrackSample]:
    root = dataset_root or problem_config.DATASET_ROOT
    return load_split(
        root,
        problem_config.DATASET_CONFIG["train_split"],
        n_samples or problem_config.DATASET_CONFIG["train_n_samples"],
        seed=problem_config.DATASET_CONFIG["seed"],
    )


def _coerce_pred_mask(tool_output: dict, shape: tuple[int, int]) -> np.ndarray:
    pred_mask = np.asarray(tool_output.get("pred_mask", []), dtype=bool)
    if pred_mask.ndim != 2 or pred_mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    return pred_mask


def _zero_metrics() -> dict[str, float]:
    return {"dice": 0.0, "dice_strict": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}


def evaluate_prompt_on_samples(
    sam_prompt: str,
    samples: list[CrackSample],
) -> dict[str, float]:
    """Run SAM3 via fixed CARL chain and compute aggregate mask metrics."""
    per_sample_metrics: list[dict[str, float]] = []
    success_flags: list[float] = []
    latencies: list[float] = []

    for sample in samples:
        gt_mask = load_ground_truth_mask(sample.mask_path)
        tool_output = sam3_segment_cracks(str(sample.image_path), sam_prompt)
        pred_mask = _coerce_pred_mask(tool_output, gt_mask.shape)
        # Only a genuine tool error scores zero; a clean run that predicts an
        # empty mask is correct for empty samples (Dice 1.0 for empty/empty).
        if tool_output.get("error"):
            sample_metrics = _zero_metrics()
        else:
            sample_metrics = compute_mask_metrics(pred_mask, gt_mask)

        per_sample_metrics.append(sample_metrics)
        success_flags.append(1.0 if tool_output.get("success", False) else 0.0)
        latencies.append(float(tool_output.get("latency_sec", 0.0)))

    aggregated = aggregate_metrics(per_sample_metrics)
    aggregated["success_rate"] = mean(success_flags) if success_flags else 0.0
    aggregated["avg_latency_sec"] = mean(latencies) if latencies else 0.0
    return aggregated


def evaluate_prompt_with_patch_stitch(
    sam_prompt: str,
    samples: list[CrackSample],
) -> dict[str, float]:
    """Run SAM3 on vertical patches and score stitched full-image masks."""
    per_sample_metrics: list[dict[str, float]] = []
    success_flags: list[float] = []
    latencies: list[float] = []

    for sample in samples:
        gt_mask = load_ground_truth_mask(sample.mask_path)
        stitched = np.zeros_like(gt_mask, dtype=bool)
        any_success = False
        sample_latency = 0.0

        patches = build_patch_samples([sample], materialize=True)
        any_ok = False
        for patch in patches:
            tool_output = sam3_segment_cracks(str(patch.image_path), sam_prompt)
            patch_shape = (patch.y1 - patch.y0, patch.x1 - patch.x0)
            pred_patch = _coerce_pred_mask(tool_output, patch_shape)
            stitched[patch.y0 : patch.y1, patch.x0 : patch.x1] |= pred_patch
            if not tool_output.get("error"):
                any_ok = True
            any_success = any_success or bool(tool_output.get("success", False))
            sample_latency += float(tool_output.get("latency_sec", 0.0))

        # Zero only if every patch errored; an empty stitched mask is a valid
        # prediction (Dice 1.0 when the full-image GT is also empty).
        if not any_ok:
            sample_metrics = _zero_metrics()
        else:
            sample_metrics = compute_mask_metrics(stitched, gt_mask)
        per_sample_metrics.append(sample_metrics)
        success_flags.append(1.0 if any_success else 0.0)
        latencies.append(sample_latency)

    aggregated = aggregate_metrics(per_sample_metrics)
    aggregated["success_rate"] = mean(success_flags) if success_flags else 0.0
    aggregated["avg_latency_sec"] = mean(latencies) if latencies else 0.0
    return aggregated


def validate(sam_prompt: str) -> dict[str, float]:
    """Validate evolved SAM3 prompt and compute training fitness."""
    validate_sam3_prompt(
        sam_prompt,
        max_length=problem_config.PROMPT_CONFIG["max_length"],
        forbidden_substrings=problem_config.PROMPT_CONFIG["forbidden_substrings"],
    )

    samples = load_train_samples()
    mode = os.environ.get("COREFRAC_EVAL_MODE", problem_config.EVAL_MODE)
    if mode in {"patch", "patch_stitch"}:
        metrics = evaluate_prompt_with_patch_stitch(sam_prompt.strip(), samples)
    else:
        metrics = evaluate_prompt_on_samples(sam_prompt.strip(), samples)

    return {
        "fitness": metrics["fitness"],
        "dice_strict": metrics["dice_strict"],
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "success_rate": metrics["success_rate"],
        "avg_latency_sec": metrics["avg_latency_sec"],
        "is_valid": 1,
    }
