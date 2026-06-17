"""Dataset and patch utilities for geological core fracture images."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from problems.prompts.corefrac import (
    config as problem_config,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CrackSample:
    """One color core image and its fracture mask."""

    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PatchSample(CrackSample):
    """A cached crop with coordinates in its source full image."""

    source_sample_id: str
    source_image_path: Path
    source_mask_path: Path
    x0: int
    y0: int
    x1: int
    y1: int
    is_positive: bool


@dataclass(frozen=True)
class PatchWindow:
    """Patch window coordinates for an image."""

    x0: int
    y0: int
    x1: int
    y1: int
    is_positive: bool
    mask_pixels: int


def _normalize_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_original", "_binary_mask", "_mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _source_group(sample_id: str) -> str:
    """Return the source-core id a patch was cropped from, for source-aware splits.

    The released dataset encodes the source core directly in the sample id as
    ``..._core_XXXX__y...``; grouping on that id keeps every crop of one source
    column in a single split. NOTE: this grouping is used by
    ``_assign_grouped_splits`` only. The released ``corefrac_split_manifest.json``
    was instead built by per-patch farthest-point sampling and is *not* grouped by
    source, so crops from one column can span splits; pass no manifest (or rebuild
    one with this grouping) for a source-aware split.
    """
    core_match = re.search(r"core_\d+", sample_id)
    if core_match:
        return core_match.group(0)
    # Fallback for ad-hoc ids: strip a leading index/label and trailing crop tag.
    stem = re.sub(r"__y\d+_\d+$", "", sample_id)
    return stem


def _find_images(root: Path) -> dict[str, Path]:
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    indexed: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            indexed[_normalize_stem(path)] = path.resolve()
    return indexed


def _find_masks(root: Path) -> dict[str, Path]:
    masks_dir = root / "binary_masks"
    if not masks_dir.is_dir():
        masks_dir = root / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks directory: {root}/(binary_)masks")
    indexed: dict[str, Path] = {}
    for path in sorted(masks_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
            indexed[_normalize_stem(path)] = path.resolve()
    return indexed


def load_all_samples(root: Path | None = None) -> list[CrackSample]:
    """Load all matched CoreFrac image/mask pairs."""
    root = root or problem_config.DATASET_ROOT
    image_index = _find_images(root)
    mask_index = _find_masks(root)

    samples: list[CrackSample] = []
    for sample_id in sorted(set(image_index) & set(mask_index)):
        samples.append(
            CrackSample(
                sample_id=sample_id,
                image_path=image_index[sample_id],
                mask_path=mask_index[sample_id],
            )
        )
    if not samples:
        raise FileNotFoundError(
            f"No matched CoreFrac image/mask pairs under {root}. "
            "Expected images/* and binary_masks/*_binary_mask.png."
        )
    return samples


def _load_manifest(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in data:
        data = data["splits"]
    return {split: [str(item) for item in data.get(split, [])] for split in SPLITS}


def _assign_grouped_splits(
    samples: list[CrackSample], seed: int
) -> dict[str, list[str]]:
    groups: dict[str, list[CrackSample]] = {}
    for sample in samples:
        groups.setdefault(_source_group(sample.sample_id), []).append(sample)

    rng = random.Random(seed)
    ordered_groups = sorted(groups.items())
    rng.shuffle(ordered_groups)

    total = len(samples)
    train_target = round(total * problem_config.DATASET_CONFIG["train_fraction"])
    val_target = round(total * problem_config.DATASET_CONFIG["val_fraction"])

    split_items: dict[str, list[str]] = {split: [] for split in SPLITS}
    for _group, group_samples in ordered_groups:
        train_gap = train_target - len(split_items["train"])
        val_gap = val_target - len(split_items["val"])
        if train_gap > 0:
            split = "train"
        elif val_gap > 0:
            split = "val"
        else:
            split = "test"
        split_items[split].extend(
            sample.sample_id
            for sample in sorted(group_samples, key=lambda s: s.sample_id)
        )

    for split in SPLITS:
        split_items[split].sort()
    return split_items


def write_split_manifest(
    path: Path,
    root: Path | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Write a deterministic split manifest for reproducible experiments."""
    root = root or problem_config.DATASET_ROOT
    seed = problem_config.DATASET_CONFIG["seed"] if seed is None else seed
    samples = load_all_samples(root)
    splits = _assign_grouped_splits(samples, seed)
    payload = {"dataset_root": str(root.resolve()), "seed": seed, "splits": splits}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return splits


def get_split_map(
    root: Path | None = None, seed: int | None = None
) -> dict[str, list[str]]:
    """Return split assignments from manifest or deterministic grouping."""
    manifest = problem_config.DATASET_CONFIG.get("split_manifest")
    if manifest:
        return _load_manifest(Path(manifest))
    samples = load_all_samples(root)
    split_seed = problem_config.DATASET_CONFIG["seed"] if seed is None else seed
    return _assign_grouped_splits(samples, split_seed)


def load_split(
    root: Path | None = None,
    split: str = "train",
    n_samples: int | None = None,
    *,
    seed: int | None = None,
) -> list[CrackSample]:
    """Load one deterministic CoreFrac split."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}")
    root = root or problem_config.DATASET_ROOT
    all_samples = {sample.sample_id: sample for sample in load_all_samples(root)}
    split_map = get_split_map(root, seed)
    samples = [
        all_samples[sample_id]
        for sample_id in split_map.get(split, [])
        if sample_id in all_samples
    ]
    if n_samples is not None and n_samples < len(samples):
        samples = samples[:n_samples]
    if not samples:
        raise FileNotFoundError(f"No samples found for split {split!r} under {root}")
    return samples


def load_ground_truth_mask(
    mask_path: str | Path, threshold: int | None = None
) -> np.ndarray:
    """Load a soft grayscale mask as a bool array."""
    threshold = (
        problem_config.DATASET_CONFIG["mask_threshold"]
        if threshold is None
        else threshold
    )
    with Image.open(mask_path) as img:
        gray = np.asarray(img.convert("L"))
    return gray > threshold


def iter_vertical_windows(
    mask: np.ndarray, height: int, overlap: int
) -> list[PatchWindow]:
    """Create vertical windows spanning the full image width."""
    image_height, image_width = mask.shape
    if height <= 0:
        raise ValueError("Patch height must be positive")
    if overlap < 0 or overlap >= height:
        raise ValueError("Patch overlap must be >= 0 and < patch height")

    if image_height <= height:
        starts = [0]
    else:
        stride = height - overlap
        starts = list(range(0, image_height - height + 1, stride))
        last = image_height - height
        if starts[-1] != last:
            starts.append(last)

    windows: list[PatchWindow] = []
    for y0 in starts:
        y1 = min(image_height, y0 + height)
        patch_mask = mask[y0:y1, :]
        mask_pixels = int(patch_mask.sum())
        windows.append(
            PatchWindow(
                x0=0,
                y0=int(y0),
                x1=int(image_width),
                y1=int(y1),
                is_positive=mask_pixels
                >= problem_config.PATCH_CONFIG["positive_min_pixels"],
                mask_pixels=mask_pixels,
            )
        )
    return windows


def _select_windows(windows: list[PatchWindow], seed_key: str) -> list[PatchWindow]:
    positives = [window for window in windows if window.is_positive]
    empties = [
        window
        for window in windows
        if not window.is_positive
        and window.mask_pixels <= problem_config.PATCH_CONFIG["empty_max_pixels"]
    ]
    max_total = problem_config.PATCH_CONFIG["max_patches_per_image"]
    if max_total <= 0:
        return sorted(windows, key=lambda w: (w.y0, w.x0))

    budget = min(max_total, len(windows))
    target_pos = max(1, round(budget * problem_config.PATCH_CONFIG["positive_ratio"]))
    target_empty = budget - target_pos
    rng_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    rng.shuffle(positives)
    rng.shuffle(empties)
    selected_pos = positives[:target_pos]
    if len(selected_pos) < target_pos:
        target_empty = min(target_empty, max_total - len(selected_pos))
    selected = selected_pos + empties[:target_empty]
    return sorted(selected, key=lambda w: (w.y0, w.x0))


def patch_cache_path(
    sample: CrackSample, window: PatchWindow, cache_dir: Path | None = None
) -> Path:
    cache_dir = cache_dir or problem_config.PATCH_CONFIG["cache_dir"]
    name = f"{sample.sample_id}__y{window.y0:05d}_{window.y1:05d}.png"
    return cache_dir / sample.sample_id / name


def materialize_patch(
    sample: CrackSample,
    window: PatchWindow,
    cache_dir: Path | None = None,
) -> Path:
    """Write an image crop for SAM3 if it is not already cached."""
    out = patch_cache_path(sample, window, cache_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.image_path) as img:
        image = img.convert("RGB")
        crop = image.crop((window.x0, window.y0, window.x1, window.y1))
        crop.save(out)
    return out


def build_patch_samples(
    samples: Iterable[CrackSample],
    *,
    materialize: bool = True,
    cache_dir: Path | None = None,
) -> list[PatchSample]:
    """Build balanced vertical patch samples from full images."""
    patch_samples: list[PatchSample] = []
    for sample in samples:
        mask = load_ground_truth_mask(sample.mask_path)
        windows = iter_vertical_windows(
            mask,
            height=problem_config.PATCH_CONFIG["height"],
            overlap=problem_config.PATCH_CONFIG["overlap"],
        )
        for window in _select_windows(windows, sample.sample_id):
            image_path = (
                materialize_patch(sample, window, cache_dir)
                if materialize
                else patch_cache_path(sample, window, cache_dir)
            )
            patch_samples.append(
                PatchSample(
                    sample_id=f"{sample.sample_id}__y{window.y0}_{window.y1}",
                    image_path=image_path.resolve(),
                    mask_path=sample.mask_path.resolve(),
                    source_sample_id=sample.sample_id,
                    source_image_path=sample.image_path.resolve(),
                    source_mask_path=sample.mask_path.resolve(),
                    x0=window.x0,
                    y0=window.y0,
                    x1=window.x1,
                    y1=window.y1,
                    is_positive=window.is_positive,
                )
            )
    return patch_samples


def write_patch_manifest(path: Path, samples: Iterable[CrackSample]) -> list[dict]:
    """Write patch metadata without forcing callers to parse filenames."""
    patches = build_patch_samples(samples, materialize=True)
    payload = [asdict(patch) for patch in patches]
    for item in payload:
        for key, value in list(item.items()):
            if isinstance(value, Path):
                item[key] = str(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
