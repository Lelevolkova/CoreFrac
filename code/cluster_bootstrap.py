"""Paired source-core cluster bootstrap for CoreFrac system comparisons.

Test patches are crops of 23 source cores, so patch-level scores are correlated
within a core. Treating the 172 patches as independent understates uncertainty.
This script resamples whole source cores with replacement and reports, for a
pair of systems, each mean and the paired difference with a percentile CI.

Input is a long-format CSV of per-patch scores with the header:

    sample_id,source_core,label,system,seed,metric,value

`label` is `positive` or `empty`, `seed` identifies the training/evolution seed
(use a constant for deterministic systems), and `value` is the per-patch score
for `metric` (e.g. soft_f1 or strict_dice). Scores are averaged over seeds per
patch before resampling, matching how the paper reports three-seed means.

Usage:
    python cluster_bootstrap.py --scores results/test_scores.csv \
        --system-a evolved_prompt --system-b lora_r8 \
        --metric soft_f1 --subset positive
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBSETS = ("positive", "empty", "all")


def load_scores(
    path: Path, metric: str, subset: str, systems: tuple[str, str]
) -> dict[str, dict[str, float]]:
    """Return {system: {sample_id: seed-averaged score}} and the core of each patch."""
    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    core_of: dict[str, str] = {}

    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["metric"] != metric or row["system"] not in systems:
                continue
            if subset != "all" and row["label"] != subset:
                continue
            sample_id = row["sample_id"]
            core_of[sample_id] = row["source_core"]
            accum[(row["system"], sample_id)].append(float(row["value"]))

    scores: dict[str, dict[str, float]] = {system: {} for system in systems}
    for (system, sample_id), values in accum.items():
        scores[system][sample_id] = float(np.mean(values))

    for system in systems:
        if not scores[system]:
            raise SystemExit(f"No rows for system={system!r}, metric={metric!r}.")
    shared = set(scores[systems[0]]) & set(scores[systems[1]])
    missing = (set(scores[systems[0]]) | set(scores[systems[1]])) - shared
    if missing:
        raise SystemExit(
            f"{len(missing)} patches are not scored for both systems; "
            "the bootstrap must be paired on identical patches."
        )
    return {"scores": scores, "core_of": core_of}


def bootstrap(
    scores: dict[str, dict[str, float]],
    core_of: dict[str, str],
    systems: tuple[str, str],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    by_core: dict[str, list[str]] = defaultdict(list)
    for sample_id in scores[systems[0]]:
        by_core[core_of[sample_id]].append(sample_id)
    cores = sorted(by_core)

    # Per core: summed score and patch count, so a resample is a weighted mean
    # over whole cores rather than over patches.
    sums = {
        system: np.array(
            [sum(scores[system][s] for s in by_core[core]) for core in cores],
            dtype=float,
        )
        for system in systems
    }
    counts = np.array([len(by_core[core]) for core in cores], dtype=float)

    def weighted_mean(totals: np.ndarray, index: np.ndarray) -> np.ndarray:
        return totals[index].sum(axis=-1) / counts[index].sum(axis=-1)

    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(cores), size=(replicates, len(cores)))
    draws_a = weighted_mean(sums[systems[0]], index)
    draws_b = weighted_mean(sums[systems[1]], index)
    diff = draws_a - draws_b

    full = np.arange(len(cores))
    return {
        "n_cores": len(cores),
        "n_patches": int(counts.sum()),
        "mean_a": float(weighted_mean(sums[systems[0]], full)),
        "mean_b": float(weighted_mean(sums[systems[1]], full)),
        "diff": float(
            weighted_mean(sums[systems[0]], full) - weighted_mean(sums[systems[1]], full)
        ),
        "ci_low": float(np.percentile(diff, 2.5)),
        "ci_high": float(np.percentile(diff, 97.5)),
        "p_diff_le_zero": float(np.mean(diff <= 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="long-format CSV")
    ap.add_argument("--system-a", required=True)
    ap.add_argument("--system-b", required=True)
    ap.add_argument("--metric", default="soft_f1")
    ap.add_argument("--subset", choices=SUBSETS, default="positive")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    systems = (args.system_a, args.system_b)
    loaded = load_scores(args.scores, args.metric, args.subset, systems)
    result = bootstrap(
        loaded["scores"], loaded["core_of"], systems, args.replicates, args.seed
    )

    print(
        f"{args.metric} ({args.subset}); "
        f"{result['n_patches']} patches in {result['n_cores']} source cores; "
        f"{args.replicates} replicates"
    )
    print(f"  {args.system_a:>24}: {result['mean_a']:.3f}")
    print(f"  {args.system_b:>24}: {result['mean_b']:.3f}")
    print(
        f"  {'paired difference':>24}: {result['diff']:+.3f} "
        f"95% CI [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
    )
    print(f"  {'P(difference <= 0)':>24}: {result['p_diff_le_zero']:.4f}")


if __name__ == "__main__":
    main()
