"""Evaluate CoreFrac SAM3 prompts on val/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from problems.prompts.corefrac.config import (
    DATASET_CONFIG,
    DATASET_ROOT,
    load_baseline,
)
from problems.prompts.corefrac.utils.dataset import (
    load_split,
)
from problems.prompts.corefrac.validate import (
    evaluate_prompt_on_samples,
    evaluate_prompt_with_patch_stitch,
)
from problems.prompts.utils import RedisRunConfig, get_best_program


def load_eval_samples(
    split: str,
    n_samples: int | None = None,
    *,
    dataset_root: Path | None = None,
) -> list:
    root = dataset_root or DATASET_ROOT
    default_n = DATASET_CONFIG.get(
        f"{split}_n_samples", DATASET_CONFIG["train_n_samples"]
    )
    return load_split(
        root,
        split,
        n_samples if n_samples is not None else default_n,
        seed=DATASET_CONFIG["seed"],
    )


def _evaluate(prompt: str, samples: list, mode: str) -> dict[str, float]:
    if mode in {"patch", "patch_stitch"}:
        return evaluate_prompt_with_patch_stitch(prompt, samples)
    return evaluate_prompt_on_samples(prompt, samples)


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key in (
        "fitness",
        "dice_strict",
        "iou",
        "precision",
        "recall",
        "success_rate",
        "avg_latency_sec",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")


def test_baseline(split: str = "val", n_samples: int | None = 3, mode: str = "full"):
    prompt = load_baseline()
    print(f"Baseline SAM3 prompt: {prompt!r}\n")

    samples = load_eval_samples(split, n_samples=n_samples)
    metrics = _evaluate(prompt, samples, mode)
    _print_metrics(
        f"=== Baseline Results ({split}, {mode}, {len(samples)} samples) ===", metrics
    )
    return metrics


def test_best_prompt(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    split: str = "test",
    n_samples: int | None = None,
    mode: str = "full",
):
    config = RedisRunConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_prefix=redis_prefix,
    )
    best = get_best_program(config, fitness_col="metric_fitness", minimize=False)
    if best is None:
        print("No programs found in Redis")
        return None

    print(f"Best program ID: {best['id']}")
    print(f"Training fitness (Dice): {best['fitness']:.4f}")
    print(f"Code:\n{best['code']}\n")

    exec_globals: dict = {}
    exec(best["code"], exec_globals)
    prompt = exec_globals["entrypoint"]()

    samples = load_eval_samples(split, n_samples=n_samples)
    metrics = _evaluate(prompt, samples, mode)
    _print_metrics(f"\n=== Test Results ({split}, {mode}) ===", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CoreFrac SAM3 prompt problem")
    parser.add_argument(
        "--mode",
        choices=["baseline", "redis"],
        default="baseline",
        help="'baseline' runs seed prompt; 'redis' evaluates best evolved prompt",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["full", "patch", "patch_stitch"],
        default="full",
        help="Evaluate on full images or stitched vertical patches",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-prefix", type=str, default="")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.mode == "baseline":
        test_baseline(split=args.split, n_samples=args.n_samples, mode=args.eval_mode)
    else:
        test_best_prompt(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            split=args.split,
            n_samples=args.n_samples,
            mode=args.eval_mode,
        )
