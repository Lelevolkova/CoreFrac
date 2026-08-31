"""Configuration for geological core crack SAM3 prompt evolution."""

from __future__ import annotations

import os
from pathlib import Path

_BASE_DIR = Path(__file__).parent
_REPO_ROOT = _BASE_DIR.parents[2]


def _opt_int(name: str) -> int | None:
    """Parse an optional integer env var; absent/empty means 'no cap'."""
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None

DATASET_ROOT = Path(os.environ.get("COREFRAC_ROOT", _REPO_ROOT / "data" / "corefrac"))
SAM3_REPO = Path(os.environ.get("SAM3_REPO", _REPO_ROOT / "sam3"))
EVAL_MODE = os.environ.get("COREFRAC_EVAL_MODE", "full")

DATASET_CONFIG = {
    "train_split": "train",
    "val_split": "val",
    "test_split": "test",
    # The headline source-grouped split (99/179/172 patches from 18/25/23
    # cores) is defined by the split manifest below; these fractions are only
    # the fallback used when no manifest is provided.
    "train_fraction": float(os.environ.get("COREFRAC_TRAIN_FRACTION", "0.213")),
    "val_fraction": float(os.environ.get("COREFRAC_VAL_FRACTION", "0.396")),
    "test_fraction": float(os.environ.get("COREFRAC_TEST_FRACTION", "0.391")),
    # Default to no cap: evaluate on the full split as defined by the manifest.
    # Set COREFRAC_{TRAIN,VAL,TEST}_N_SAMPLES to subsample (evolution scored
    # candidates on a 64-patch subsample of the 96-patch train split).
    "train_n_samples": _opt_int("COREFRAC_TRAIN_N_SAMPLES"),
    "val_n_samples": _opt_int("COREFRAC_VAL_N_SAMPLES"),
    "test_n_samples": _opt_int("COREFRAC_TEST_N_SAMPLES"),
    "seed": int(os.environ.get("COREFRAC_SEED", "42")),
    "split_manifest": os.environ.get("COREFRAC_SPLIT_MANIFEST"),
    "mask_threshold": int(os.environ.get("COREFRAC_MASK_THRESHOLD", "127")),
}

PATCH_CONFIG = {
    "height": int(os.environ.get("COREFRAC_PATCH_HEIGHT", "256")),
    "overlap": int(os.environ.get("COREFRAC_PATCH_OVERLAP", "64")),
    "positive_ratio": float(os.environ.get("COREFRAC_PATCH_POSITIVE_RATIO", "0.70")),
    "positive_min_pixels": int(
        os.environ.get("COREFRAC_PATCH_POSITIVE_MIN_PIXELS", "16")
    ),
    "empty_max_pixels": int(os.environ.get("COREFRAC_PATCH_EMPTY_MAX_PIXELS", "0")),
    "max_patches_per_image": int(
        os.environ.get("COREFRAC_MAX_PATCHES_PER_IMAGE", "24")
    ),
    "cache_dir": Path(
        os.environ.get(
            "COREFRAC_PATCH_CACHE",
            str(_REPO_ROOT / ".cache" / "corefrac" / "patches"),
        )
    ),
}

SAM3_CONFIG = {
    "backend": os.environ.get("SAM3_BACKEND", "transformers"),
    "model_name": os.environ.get("SAM3_MODEL_NAME", "facebook/sam3"),
    "checkpoint_path": os.environ.get("SAM3_CHECKPOINT_PATH"),
    "load_from_hf": os.environ.get("SAM3_LOAD_FROM_HF", "false").lower()
    in {"1", "true", "yes"},
    "confidence_threshold": float(os.environ.get("SAM3_CONFIDENCE", "0.5")),
    "device": os.environ.get("SAM3_DEVICE", "cuda"),
    "allow_cpu": os.environ.get("SAM3_ALLOW_CPU", "false").lower()
    in {"1", "true", "yes"},
    "resolution": int(os.environ.get("SAM3_RESOLUTION", "1008")),
}

PROMPT_CONFIG = {
    "max_length": int(os.environ.get("SAM3_PROMPT_MAX_LENGTH", "180")),
    "forbidden_substrings": [
        "final_dataset",
        "binary_mask",
        "original.png",
        "mask_path",
        "image_path",
        "sample_id",
        "train/",
        "val/",
        "test/",
        ".jpg",
        ".jpeg",
        ".png",
    ],
}

PROMPT_BASELINES = {
    "crack500_generic": "thin dark cracks in rock surface",
    "dark_geological": "dark thin open fractures in geological drill core",
    "black_core": "black branching cracks and fissures in rock core",
    "healed_light": "light mineral filled healed fractures in rock core",
    "pale_sealed": "pale linear sealed cracks in drill core",
    "combined": "dark cracks and light healed fractures in geological core",
    "visible_core_hairline": (
        "visible cracks in rock core sample, including thin and branching fractures, "
        "with varying contrast against the rock matrix, high-contrast hairline"
    ),
}


def load_baseline() -> str:
    """Load baseline SAM3 text prompt from initial_programs/baseline.py."""
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals: dict = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
