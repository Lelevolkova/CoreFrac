"""Build a deterministic, diversity-aware split manifest for fast evolution.

Selects a stratified ~96-patch training subset (label ratio matching the source
dataset) from the materialized diverse patch dataset using greedy farthest-point
sampling over the per-patch feature vectors already stored in the manifest. The
remaining patches are split into val/test. Evolution scores candidates on the
small `train` subset; champions are re-validated on all 450 patches separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

FEATURE_KEYS = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "edge_density",
    "laplacian_var",
    "crack_ratio",
    "crack_components",
    "crack_bbox_fill",
    "crack_y_span",
    "crack_x_span",
]


def _feature_matrix(rows: list[dict]) -> np.ndarray:
    mat = np.array(
        [[float(r["features"].get(k, 0.0)) for k in FEATURE_KEYS] for r in rows],
        dtype=np.float64,
    )
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std == 0] = 1.0
    return (mat - mean) / std


def _farthest_point_order(feats: np.ndarray, seed: int) -> list[int]:
    """Greedy farthest-point ordering; deterministic given seed."""
    n = feats.shape[0]
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n))
    selected = [start]
    min_d = np.linalg.norm(feats - feats[start], axis=1)
    while len(selected) < n:
        nxt = int(np.argmax(min_d))
        if nxt in selected:
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = remaining[0]
        selected.append(nxt)
        d = np.linalg.norm(feats - feats[nxt], axis=1)
        min_d = np.minimum(min_d, d)
    return selected


def _stratified_pick(
    rows: list[dict], label: str, n_pick: int, seed: int
) -> tuple[list[str], list[str]]:
    group = [r for r in rows if r["label"] == label]
    if not group:
        return [], []
    feats = _feature_matrix(group)
    order = _farthest_point_order(feats, seed)
    ids = [group[i]["sample_id"] for i in order]
    return ids[:n_pick], ids[n_pick:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("COREFRAC_ROOT", "data/corefrac/patches"),
        help="CoreFrac patch dataset root (images/, masks/, manifest.jsonl)",
    )
    parser.add_argument("--train-total", type=int, default=96)
    parser.add_argument("--positive-ratio", type=float, default=0.80)
    parser.add_argument("--val-frac-of-rest", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ds = Path(args.dataset_dir)
    rows = [
        json.loads(line)
        for line in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    n_pos = round(args.train_total * args.positive_ratio)
    n_emp = args.train_total - n_pos

    train_pos, rest_pos = _stratified_pick(rows, "positive", n_pos, args.seed)
    train_emp, rest_emp = _stratified_pick(rows, "empty", n_emp, args.seed + 1)

    def split_rest(rest: list[str], seed: int) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(rest))
        rng.shuffle(idx)
        cut = round(len(rest) * args.val_frac_of_rest)
        val = sorted(rest[i] for i in idx[:cut])
        test = sorted(rest[i] for i in idx[cut:])
        return val, test

    val_pos, test_pos = split_rest(rest_pos, args.seed + 2)
    val_emp, test_emp = split_rest(rest_emp, args.seed + 3)

    splits = {
        "train": sorted(train_pos + train_emp),
        "val": sorted(val_pos + val_emp),
        "test": sorted(test_pos + test_emp),
    }
    payload = {
        "dataset_root": str(ds.resolve()),
        "seed": args.seed,
        "selection": "farthest_point_stratified",
        "counts": {
            "train": {"positive": len(train_pos), "empty": len(train_emp)},
            "val": {"positive": len(val_pos), "empty": len(val_emp)},
            "test": {"positive": len(test_pos), "empty": len(test_emp)},
        },
        "splits": splits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
