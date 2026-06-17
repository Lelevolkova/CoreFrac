"""Verify CoreFrac dataset + SAM3 + CARL setup."""

from __future__ import annotations

import os
from pathlib import Path

from problems.prompts.corefrac import config
from problems.prompts.corefrac.utils.dataset import (
    load_all_samples,
    load_split,
)


def check_dataset(root: Path) -> tuple[bool, str]:
    try:
        samples = load_all_samples(root)
        split_counts = {
            split: len(load_split(root, split)) for split in ("train", "val", "test")
        }
    except Exception as exc:
        return False, f"Dataset check failed: {exc}"
    return True, f"{len(samples)} pairs at {root}; splits={split_counts}"


def check_cuda() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch import failed: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available; production CoreFrac runs are GPU-only"
    return True, f"CUDA available: {torch.cuda.device_count()} device(s)"


def check_carl() -> tuple[bool, str]:
    try:
        import mmar_carl  # noqa: F401
    except Exception as exc:
        return False, f"CARL unavailable: {exc}"
    return True, "CARL import ok"


def main() -> int:
    root = Path(os.environ.get("COREFRAC_ROOT", str(config.DATASET_ROOT)))
    checks = [
        ("dataset", check_dataset(root)),
        ("cuda", check_cuda()),
        ("carl", check_carl()),
    ]
    print("Setup verification for prompts/corefrac\n")
    ok = True
    for name, (passed, message) in checks:
        ok = ok and passed
        print(f"[{name}] {'OK' if passed else 'FAIL'}: {message}")
    if not ok:
        print("\nFix failed checks before long SAM3 evolution runs.")
        return 1
    print("\nSmoke command:")
    print(
        "  python problems/prompts/corefrac/test.py --mode baseline --split val --n-samples 3"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
