"""Soft (tolerance) mask metrics for thin-crack segmentation.

Plain pixel Dice has a low ceiling on hair-thin curvilinear cracks: a perfect
detection shifted by a single pixel already drops Dice to ~0.62 (measured on this
dataset). This package therefore scores with a **tolerance-F1** (CrackForest /
DeepCrack style): a predicted pixel counts as correct if it lies within ``tol``
pixels of any ground-truth crack pixel, and a GT pixel counts as recovered if it
lies within ``tol`` pixels of any predicted pixel.

  precision_tol = |pred pixels within tol of GT| / |pred|
  recall_tol    = |GT pixels within tol of pred| / |GT|
  f1_tol        = 2 * p * r / (p + r)      <-- primary fitness ("dice")

Strict pixel Dice/IoU are still reported (``dice_strict``, ``iou``) so progress
can be read on both the soft and the strict scale.
"""

from __future__ import annotations

import os

import numpy as np

# Default slack in pixels; overridable so the same code can run stricter/looser.
DEFAULT_TOLERANCE = int(os.environ.get("COREFRAC_METRIC_TOLERANCE", "2"))


def _to_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    return mask > 0


def _tolerance_pr(pred: np.ndarray, gt: np.ndarray, tol: int) -> tuple[float, float]:
    """Return (precision_tol, recall_tol) using a Euclidean distance transform."""
    from scipy.ndimage import distance_transform_edt

    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    if pred_sum == 0.0 or gt_sum == 0.0:
        return 0.0, 0.0

    # distance from each pixel to the nearest GT (resp. pred) crack pixel.
    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)
    precision = float((dist_to_gt[pred] <= tol).sum()) / pred_sum
    recall = float((dist_to_pred[gt] <= tol).sum()) / gt_sum
    return precision, recall


def compute_mask_metrics(
    pred: np.ndarray, gt: np.ndarray, tol: int | None = None
) -> dict[str, float]:
    """Compute tolerance-F1 (primary) plus strict pixel Dice/IoU for two masks."""
    pred_b = _to_bool_mask(pred)
    gt_b = _to_bool_mask(gt)

    if pred_b.shape != gt_b.shape:
        raise ValueError(f"Mask shape mismatch: pred={pred_b.shape}, gt={gt_b.shape}")

    tol = DEFAULT_TOLERANCE if tol is None else tol

    pred_sum = float(pred_b.sum())
    gt_sum = float(gt_b.sum())

    # Empty-vs-empty is a correct prediction (score 1.0); empty-vs-nonempty is 0.0.
    if pred_sum == 0.0 and gt_sum == 0.0:
        return {
            "dice": 1.0,
            "dice_strict": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        }

    intersection = float(np.logical_and(pred_b, gt_b).sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    dice_strict = (2.0 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 0.0
    iou = (intersection / union) if union > 0 else 0.0

    p_tol, r_tol = _tolerance_pr(pred_b, gt_b, tol)
    f1_tol = (2.0 * p_tol * r_tol / (p_tol + r_tol)) if (p_tol + r_tol) > 0 else 0.0

    return {
        "dice": f1_tol,          # primary fitness component (soft tolerance-F1)
        "dice_strict": dice_strict,  # strict pixel Dice, for reference
        "iou": iou,                  # strict pixel IoU, for reference
        "precision": p_tol,          # tolerance precision
        "recall": r_tol,             # tolerance recall
    }


def aggregate_metrics(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dicts across samples. Fitness == mean soft tolerance-F1."""
    if not per_sample:
        return {
            "fitness": 0.0,
            "dice_strict": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    keys = ("dice", "dice_strict", "iou", "precision", "recall")
    totals = {key: 0.0 for key in keys}
    for sample_metrics in per_sample:
        for key in keys:
            totals[key] += sample_metrics.get(key, 0.0)

    n = len(per_sample)
    return {
        "fitness": totals["dice"] / n,
        "dice_strict": totals["dice_strict"] / n,
        "iou": totals["iou"] / n,
        "precision": totals["precision"] / n,
        "recall": totals["recall"] / n,
    }
