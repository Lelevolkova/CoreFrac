#!/usr/bin/env python3
"""Train a PEFT LoRA adapter for Hugging Face ``Sam3Model`` on CoreFrac.

SAM3 does not expose a built-in binary-mask training loss. For the configured
BCE-with-logits + soft-Dice objective, this example constructs one binary
union logit per pixel from the instance-mask logits and query/presence scores.
At threshold 0.5, ``max_q(min(mask_logit_q, score_logit_q)) > 0`` implements the
same AND-then-OR decision as thresholding accepted instances and taking their
union. This is a differentiable (piecewise) surrogate, not Meta's native SAM3
matching loss.

Example (two A100 GPUs), from the repository root:

    accelerate launch --num_processes 2 code/train_sam3_lora.py \
      --config configs/lora/corefrac_sam3_lora_r8_grouped.yaml \
      --manifest dataset/patches/splits_grouped/corefrac_grouped_split_manifest.json \
      --dataset-root dataset/patches \
      --train-subset dataset/patches/splits_grouped/adaptation_pool_96.json \
      --prompt "<the fixed LoRA training prompt>" \
      --seed 0

The subset JSON may be a list of sample IDs, ``{"sample_ids": [...]}``, or a
manifest-shaped object with ``{"splits": {"train": [...]}}``.

Runtime dependencies: PyTorch, a Transformers release containing ``Sam3Model``,
PEFT, Accelerate, PyYAML, Pillow, SciPy, and (for the YAML default) flash-attn.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance


MODEL_INPUT_KEYS = {
    "pixel_values",
    "input_ids",
    "attention_mask",
    "vision_embeds",
    "text_embeds",
}


@dataclass(frozen=True)
class PatchRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    label: str
    source_core: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--train-subset", type=Path)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--attention-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        help="Overrides model.use_flash_attention from the YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/config and build the PEFT model, then stop.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML: pip install pyyaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_placeholder(value: Any) -> bool:
    return not value or (isinstance(value, str) and value.startswith("<") and value.endswith(">"))


def resolve_path(cli_value: Path | None, config_value: Any, name: str) -> Path:
    raw = cli_value if cli_value is not None else config_value
    if is_placeholder(str(raw) if raw is not None else raw):
        raise ValueError(f"{name} is unresolved; provide the corresponding CLI argument")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_subset_ids(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, list):
        ids = payload
    elif isinstance(payload, dict) and isinstance(payload.get("sample_ids"), list):
        ids = payload["sample_ids"]
    elif isinstance(payload, dict) and isinstance(payload.get("splits"), dict):
        ids = payload["splits"].get("train")
    else:
        ids = None
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError(
            f"Unsupported subset schema in {path}; expected list[str], "
            "{'sample_ids': [...]}, or {'splits': {'train': [...]}}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in adaptation subset: {path}")
    return ids


def load_patch_index(dataset_root: Path) -> dict[str, PatchRecord]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing patch manifest: {manifest_path}")

    records: dict[str, PatchRecord] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in records:
            raise ValueError(f"Duplicate sample_id {sample_id!r} at line {line_number}")
        label = str(row["label"])
        if label not in {"positive", "empty"}:
            raise ValueError(f"Unexpected label {label!r} for {sample_id}")
        records[sample_id] = PatchRecord(
            sample_id=sample_id,
            image_path=(dataset_root / row["image_path"]).resolve(),
            mask_path=(dataset_root / row["mask_path"]).resolve(),
            label=label,
            source_core=str(row["source_core"]),
        )
    return records


def count_labels(records: Iterable[PatchRecord]) -> dict[str, int]:
    counts = {"positive": 0, "empty": 0}
    for record in records:
        counts[record.label] += 1
    return counts


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def resolve_records(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[list[PatchRecord], list[PatchRecord], Path, Path, Path]:
    data_cfg = config["data"]
    manifest_path = resolve_path(args.manifest, data_cfg.get("manifest_path"), "grouped manifest")
    grouped = read_json(manifest_path)
    if not isinstance(grouped, dict) or not isinstance(grouped.get("splits"), dict):
        raise ValueError(f"Grouped manifest has no splits mapping: {manifest_path}")

    dataset_value = args.dataset_root or data_cfg.get("dataset_root")
    if is_placeholder(str(dataset_value) if dataset_value is not None else dataset_value):
        dataset_value = grouped.get("dataset_root")
    dataset_root = resolve_path(
        Path(dataset_value) if dataset_value is not None else None,
        None,
        "dataset root",
    )
    subset_path = resolve_path(
        args.train_subset,
        data_cfg["splits"]["train"].get("adaptation_subset_path"),
        "fixed 96-patch subset",
    )

    index = load_patch_index(dataset_root)
    split_ids = {name: [str(item) for item in grouped["splits"].get(name, [])] for name in ("train", "val", "test")}
    all_ids = [item for ids in split_ids.values() for item in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Grouped manifest contains a sample in more than one split")
    missing = sorted(set(all_ids) - set(index))
    if missing:
        raise ValueError(f"Grouped manifest references unknown samples: {missing[:5]}")

    train_cfg = data_cfg["splits"]["train"]
    val_cfg = data_cfg["splits"]["validation"]
    test_cfg = data_cfg["splits"]["test"]
    require_equal(len(split_ids["train"]), int(train_cfg["manifest_size"]), "grouped train size")
    require_equal(len(split_ids["val"]), int(val_cfg["patches"]), "grouped validation size")
    require_equal(len(split_ids["test"]), int(test_cfg["patches"]), "grouped test size")

    train_subset = load_subset_ids(subset_path)
    require_equal(len(train_subset), int(train_cfg["selected_patches"]), "adaptation subset size")
    outside_train = sorted(set(train_subset) - set(split_ids["train"]))
    if outside_train:
        raise ValueError(f"Adaptation subset contains IDs outside grouped train: {outside_train[:5]}")

    train_records = [index[item] for item in train_subset]
    val_records = [index[item] for item in split_ids["val"]]
    test_records = [index[item] for item in split_ids["test"]]
    train_counts = count_labels(train_records)
    val_counts = count_labels(val_records)
    test_counts = count_labels(test_records)
    require_equal(train_counts["positive"], int(train_cfg["expected_positive"]), "train positives")
    require_equal(train_counts["empty"], int(train_cfg["expected_empty"]), "train empty patches")
    require_equal(val_counts["positive"], int(val_cfg["expected_positive"]), "validation positives")
    require_equal(val_counts["empty"], int(val_cfg["expected_empty"]), "validation empty patches")
    require_equal(test_counts["positive"], int(test_cfg["expected_positive"]), "test positives")
    require_equal(test_counts["empty"], int(test_cfg["expected_empty"]), "test empty patches")

    split_records = {
        "train": [index[item] for item in split_ids["train"]],
        "val": val_records,
        "test": test_records,
    }
    split_cores = {name: {record.source_core for record in rows} for name, rows in split_records.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_cores[left] & split_cores[right]
        if overlap:
            raise ValueError(f"Source-core overlap between {left} and {right}: {sorted(overlap)}")
    require_equal(len(split_cores["train"]), int(train_cfg["source_cores"]), "train source cores")
    require_equal(len(split_cores["val"]), int(val_cfg["source_cores"]), "validation source cores")
    require_equal(len(split_cores["test"]), int(test_cfg["source_cores"]), "test source cores")

    for record in train_records + val_records:
        if not record.image_path.is_file() or not record.mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask for {record.sample_id}")
    for record in train_records:
        with Image.open(record.mask_path) as image:
            actually_positive = bool(np.asarray(image.convert("L"), dtype=np.uint8).any())
        if actually_positive != (record.label == "positive"):
            raise ValueError(f"Manifest/mask label mismatch for {record.sample_id}")

    return train_records, val_records, dataset_root, manifest_path, subset_path


class CoreFracDataset:
    def __init__(self, records: list[PatchRecord], augmentation: dict[str, Any] | None):
        self.records = records
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(record.mask_path) as mask_file:
            mask = mask_file.convert("L").point(lambda value: 255 if value > 0 else 0)
        if self.augmentation:
            image, mask = augment_pair(image, mask, self.augmentation)
        return {"record": record, "image": image, "mask": mask}


def augment_pair(
    image: Image.Image, mask: Image.Image, config: dict[str, Any]
) -> tuple[Image.Image, Image.Image]:
    if random.random() < float(config.get("horizontal_flip_probability", 0.0)):
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < float(config.get("vertical_flip_probability", 0.0)):
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < float(config.get("random_brightness_probability", 0.0)):
        limit = float(config.get("random_brightness_limit", 0.0))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(1.0 - limit, 1.0 + limit))
    if random.random() < float(config.get("random_contrast_probability", 0.0)):
        limit = float(config.get("random_contrast_limit", 0.0))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(1.0 - limit, 1.0 + limit))
    return image, mask


class Sam3Collator:
    def __init__(self, processor: Any, prompt: str):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images = [item["image"] for item in items]
        masks = [item["mask"] for item in items]
        encoded = self.processor(
            images=images,
            text=[self.prompt] * len(items),
            segmentation_maps=masks,
            return_tensors="pt",
        )
        label_key = "labels" if "labels" in encoded else "segmentation_maps"
        if label_key not in encoded:
            raise KeyError(
                "Sam3Processor did not return processed masks as labels; "
                f"available keys: {sorted(encoded.keys())}"
            )
        labels = encoded.pop(label_key)
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        encoded["labels"] = (labels > 0).to(dtype=torch.float32)
        encoded["ground_truth_masks"] = [
            torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True) > 0) for mask in masks
        ]
        encoded["sample_ids"] = [item["record"].sample_id for item in items]
        encoded["is_positive"] = torch.tensor(
            [item["record"].label == "positive" for item in items], dtype=torch.bool
        )
        return encoded


def seed_worker(worker_id: int) -> None:
    del worker_id
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_target_modules(model: Any, config: dict[str, Any]) -> list[str]:
    import torch

    suffixes = tuple(str(item) for item in config["target_module_suffixes"])
    includes = [re.compile(str(pattern)) for pattern in config["include_component_patterns"]]
    excludes = [re.compile(str(pattern)) for pattern in config["exclude_component_patterns"]]
    selected: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if not any(pattern.search(name) for pattern in includes):
            continue
        if any(pattern.search(name) for pattern in excludes):
            continue
        selected.append(name)
    selected.sort()
    if not selected and config.get("fail_if_no_targets", True):
        raise RuntimeError("No LoRA target modules matched the YAML rules")
    return selected


def trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def build_model(config: dict[str, Any], args: argparse.Namespace) -> tuple[Any, Any, list[str], int]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Sam3Model, Sam3Processor

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    attention = args.attention_implementation
    if attention is None:
        attention = "flash_attention_2" if model_cfg.get("use_flash_attention") else "sdpa"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtypes[dtype_name],
        "attn_implementation": attention,
    }
    try:
        model = Sam3Model.from_pretrained(model_cfg["model_id"], **load_kwargs)
    except (ImportError, RuntimeError) as exc:
        if attention == "flash_attention_2":
            raise RuntimeError(
                "Could not load SAM3 with FlashAttention 2. Install flash-attn "
                "or rerun with --attention-implementation sdpa."
            ) from exc
        raise
    processor = Sam3Processor.from_pretrained(model_cfg.get("processor_id", model_cfg["model_id"]))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = select_target_modules(model, lora_cfg)
    peft_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg["bias"]),
        target_modules=targets,
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable, _total = trainable_parameters(model)
    expected = lora_cfg["expected_trainable_parameters"]
    if lora_cfg.get("fail_if_outside_parameter_range", True):
        minimum, maximum = int(expected["min"]), int(expected["max"])
        if not minimum <= trainable <= maximum:
            raise RuntimeError(
                f"Trainable parameter count {trainable:,} is outside [{minimum:,}, {maximum:,}]"
            )
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_A." not in name and ".lora_B." not in name
    ]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {unexpected[:5]}")
    return model, processor, targets, trainable


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def union_logits(outputs: Any, target_shape: tuple[int, int]) -> Any:
    """Piecewise-differentiable instance-union logit at a 0.5 threshold."""
    import torch
    import torch.nn.functional as functional

    masks = outputs.pred_masks.float()
    if masks.ndim == 5 and masks.shape[2] == 1:
        masks = masks.squeeze(2)
    if masks.ndim != 4:
        raise ValueError(f"Unexpected pred_masks shape: {tuple(masks.shape)}")
    if tuple(masks.shape[-2:]) != tuple(target_shape):
        masks = functional.interpolate(masks, size=target_shape, mode="bilinear", align_corners=False)

    query_logits = outputs.pred_logits.float()
    if query_logits.ndim == 3 and query_logits.shape[-1] == 1:
        query_logits = query_logits.squeeze(-1)
    if query_logits.ndim != 2:
        raise ValueError(f"Unexpected pred_logits shape: {tuple(query_logits.shape)}")
    score_probability = torch.sigmoid(query_logits)

    presence_logits = getattr(outputs, "presence_logits", None)
    if presence_logits is not None:
        presence_probability = torch.sigmoid(presence_logits.float())
        while presence_probability.ndim > 2 and presence_probability.shape[-1] == 1:
            presence_probability = presence_probability.squeeze(-1)
        if presence_probability.ndim == 1:
            presence_probability = presence_probability.unsqueeze(-1)
        if presence_probability.shape[-1] == 1:
            presence_probability = presence_probability.expand_as(score_probability)
        if presence_probability.shape != score_probability.shape:
            raise ValueError(
                "presence_logits cannot be broadcast to pred_logits: "
                f"{tuple(presence_probability.shape)} vs {tuple(score_probability.shape)}"
            )
        score_probability = score_probability * presence_probability

    eps = torch.finfo(torch.float32).eps
    score_logits = torch.logit(score_probability.clamp(eps, 1.0 - eps))
    accepted_instance_logits = torch.minimum(masks, score_logits[..., None, None])
    return accepted_instance_logits.amax(dim=1, keepdim=True)


def segmentation_loss(logits: Any, labels: Any, loss_config: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    weights = {str(item["name"]): float(item["weight"]) for item in loss_config["components"]}
    dims = tuple(range(1, logits.ndim))
    bce_per_example = functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    ).mean(dim=dims)
    empty = labels.sum(dim=dims) == 0
    example_weights = torch.ones_like(bce_per_example)
    example_weights[empty] = float(loss_config.get("empty_example_weight", 1.0))
    bce = (bce_per_example * example_weights).sum() / example_weights.sum()
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * labels).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + labels.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    soft_dice = ((1.0 - dice) * example_weights).sum() / example_weights.sum()
    total = (
        weights.get("binary_cross_entropy_with_logits", 0.0) * bce
        + weights.get("soft_dice", 0.0) * soft_dice
    )
    return total, {"bce": float(bce.detach()), "soft_dice": float(soft_dice.detach())}


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: int) -> dict[str, float]:
    from scipy.ndimage import distance_transform_edt

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} vs {gt.shape}")
    pred_sum, gt_sum = float(pred.sum()), float(gt.sum())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return {"soft_f1": 1.0, "strict_dice": 1.0, "strict_iou": 1.0}
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    strict_dice = 2.0 * intersection / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0
    strict_iou = intersection / union if union else 0.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        soft_f1 = 0.0
    else:
        precision = float((distance_transform_edt(~gt)[pred] <= tolerance).sum()) / pred_sum
        recall = float((distance_transform_edt(~pred)[gt] <= tolerance).sum()) / gt_sum
        soft_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"soft_f1": soft_f1, "strict_dice": strict_dice, "strict_iou": strict_iou}


def union_postprocessed_masks(
    result: dict[str, Any], shape: tuple[int, int], mask_threshold: float
) -> np.ndarray:
    import torch

    masks = result.get("masks")
    if masks is None:
        return np.zeros(shape, dtype=bool)
    if isinstance(masks, torch.Tensor):
        array = masks.detach().cpu().numpy()
    else:
        array = np.asarray(masks)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Unexpected postprocessed mask shape: {array.shape}")
    if array.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    union = np.any(array > mask_threshold, axis=0)
    if union.shape != shape:
        union = np.asarray(
            Image.fromarray(union.astype(np.uint8) * 255).resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
        ) > 0
    return union


def evaluate(
    model: Any,
    processor: Any,
    loader: Any,
    accelerator: Any,
    config: dict[str, Any],
) -> dict[str, float]:
    import torch

    model.eval()
    metric_cfg = config["metric_definition"]
    model_cfg = config["model"]
    gathered_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(**model_inputs(batch))
            original_sizes = batch.get("original_sizes")
            if original_sizes is None:
                target_sizes = [list(mask.shape) for mask in batch["ground_truth_masks"]]
            else:
                target_sizes = original_sizes.detach().cpu().tolist()
            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=float(model_cfg["confidence_threshold"]),
                mask_threshold=float(model_cfg["mask_threshold"]),
                target_sizes=target_sizes,
            )
            local_rows = []
            for index, result in enumerate(processed):
                ground_truth = batch["ground_truth_masks"][index].detach().cpu().numpy().astype(bool)
                prediction = union_postprocessed_masks(
                    result,
                    ground_truth.shape,
                    mask_threshold=float(model_cfg["mask_threshold"]),
                )
                metrics = mask_metrics(
                    prediction,
                    ground_truth,
                    tolerance=int(metric_cfg["soft_f1_tolerance_pixels"]),
                )
                positive = bool(batch["is_positive"][index].item())
                local_rows.append(
                    [
                        metrics["soft_f1"],
                        metrics["strict_dice"],
                        metrics["strict_iou"],
                        float(positive),
                        float((not positive) and prediction.any()),
                    ]
                )
            row_tensor = torch.tensor(local_rows, dtype=torch.float64, device=accelerator.device)
            gathered_rows.append(accelerator.gather_for_metrics(row_tensor).cpu())

    rows = torch.cat(gathered_rows, dim=0)
    positive = rows[:, 3] > 0.5
    empty = ~positive
    metrics = {
        "soft_f1_all": float(rows[:, 0].mean()),
        "soft_f1_positive": float(rows[positive, 0].mean()),
        "strict_dice_positive": float(rows[positive, 1].mean()),
        "strict_iou_positive": float(rows[positive, 2].mean()),
        "empty_false_positives": int(rows[empty, 4].sum()),
        "positive_count": int(positive.sum()),
        "empty_count": int(empty.sum()),
        "sample_count": int(rows.shape[0]),
        "inference_error_count": 0,
    }
    model.train()
    return metrics


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if incumbent is None:
        return True
    selection = config["checkpoint_selection"]
    tolerance = float(selection["tie_tolerance"])
    current = float(candidate["validation"]["soft_f1_all"])
    best = float(incumbent["validation"]["soft_f1_all"])
    if current > best + tolerance:
        return True
    if current < best - tolerance:
        return False
    for tie_breaker in selection["tie_breakers"]:
        path = str(tie_breaker["metric"])
        if path == "epoch":
            left, right = candidate["epoch"], incumbent["epoch"]
        else:
            _section, metric = path.split(".", 1)
            left, right = candidate["validation"][metric], incumbent["validation"][metric]
        if left == right:
            continue
        return left > right if tie_breaker["mode"] == "max" else left < right
    return False


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    config = load_yaml(args.config.resolve())
    prompt = args.prompt or config["data"].get("training_prompt")
    if not prompt or not prompt.strip():
        raise ValueError("The fixed training prompt is required via --prompt or data.training_prompt")
    prompt = prompt.strip()

    training = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        mixed_precision=str(training["mixed_precision"]),
    )
    set_seed(args.seed, device_specific=True)
    expected_world_size = int(training["num_gpus"])
    if not args.dry_run and accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            f"YAML requires {expected_world_size} GPU processes, but Accelerate started "
            f"{accelerator.num_processes}. Use `accelerate launch --num_processes "
            f"{expected_world_size} ...`."
        )
    effective_batch_size = (
        int(training["per_device_train_batch_size"])
        * accelerator.num_processes
        * int(training["gradient_accumulation_steps"])
    )
    if not args.dry_run:
        require_equal(
            effective_batch_size,
            int(training["expected_effective_batch_size"]),
            "effective batch size",
        )
    if config["reproducibility"].get("deterministic_algorithms", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = bool(config["reproducibility"].get("cudnn_benchmark", False))

    train_records, val_records, dataset_root, manifest_path, subset_path = resolve_records(config, args)
    output_dir = (args.output_dir or Path(config["output_dir"]) / f"seed_{args.seed}").resolve()
    occupied = output_dir.exists() and any(output_dir.iterdir())
    if occupied and not args.dry_run:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a fresh --output-dir; "
            "the script will not mix or overwrite runs."
        )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    model, processor, target_modules, trainable_count = build_model(config, args)
    _trainable, total_count = trainable_parameters(model)
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "train": count_labels(train_records),
                    "validation": count_labels(val_records),
                    "target_module_count": len(target_modules),
                    "trainable_parameters": trainable_count,
                    "total_parameters_with_adapter": total_count,
                    "trainable_fraction": trainable_count / total_count,
                },
                indent=2,
            )
        )
    if args.dry_run:
        return

    train_dataset = CoreFracDataset(train_records, config["augmentation"])
    val_dataset = CoreFracDataset(val_records, None)
    collator = Sam3Collator(processor, prompt)
    generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
        "collate_fn": collator,
        "num_workers": int(training["dataloader_num_workers"]),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["per_device_train_batch_size"]),
        shuffle=bool(training["shuffle_train"]),
        drop_last=bool(training["drop_last"]),
        generator=generator,
        **common_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["per_device_eval_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common_loader,
    )

    learning_rate = args.learning_rate or float(training["learning_rate"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(training["gradient_accumulation_steps"]))
    total_updates = int(training["max_epochs"]) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * float(training["warmup_ratio"])),
        num_training_steps=total_updates,
    )

    provenance = {
        "config": config,
        "config_path": str(args.config.resolve()),
        "dataset_root": str(dataset_root),
        "grouped_manifest": str(manifest_path),
        "grouped_manifest_sha256": sha256_file(manifest_path),
        "adaptation_subset": str(subset_path),
        "adaptation_subset_sha256": sha256_file(subset_path),
        "prompt": prompt,
        "seed": args.seed,
        "learning_rate": learning_rate,
        "git_commit": git_commit(),
        "target_modules": target_modules,
        "trainable_parameters": trainable_count,
        "world_size": accelerator.num_processes,
    }
    if accelerator.is_main_process:
        write_json(output_dir / "resolved_run.json", provenance)

    selection_cfg = config["checkpoint_selection"]
    frequency = int(selection_cfg["evaluation_frequency_epochs"])
    early_cfg = selection_cfg["early_stopping"]
    best: dict[str, Any] | None = None
    evaluations_without_improvement = 0
    start_time = time.perf_counter()
    model.train()

    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_loss = 0.0
        microbatches = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**model_inputs(batch))
                logits = union_logits(outputs, tuple(batch["labels"].shape[-2:]))
                loss, _parts = segmentation_loss(logits, batch["labels"], config["loss"])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach())
            microbatches += 1

        reduced_loss = accelerator.reduce(
            torch.tensor([epoch_loss, microbatches], device=accelerator.device, dtype=torch.float64),
            reduction="sum",
        )
        train_loss = float(reduced_loss[0] / reduced_loss[1])
        if epoch % frequency != 0:
            continue

        validation = evaluate(model, processor, val_loader, accelerator, config)
        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        previous_best = best
        improved = is_better(checkpoint, best, config)
        if improved:
            best = checkpoint
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                adapter_dir = output_dir / "best_adapter"
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(adapter_dir, safe_serialization=True)
                processor.save_pretrained(adapter_dir / "processor")
                write_json(output_dir / "best_checkpoint.json", checkpoint)
        primary_delta = (
            math.inf
            if previous_best is None
            else float(checkpoint["validation"]["soft_f1_all"])
            - float(previous_best["validation"]["soft_f1_all"])
        )
        if primary_delta >= float(early_cfg["minimum_delta"]):
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1

        if accelerator.is_main_process:
            append_jsonl(output_dir / "metrics.jsonl", {**checkpoint, "is_best": improved})
            print(json.dumps({**checkpoint, "is_best": improved}, ensure_ascii=False))
        accelerator.wait_for_everyone()
        if (
            early_cfg.get("enabled", True)
            and evaluations_without_improvement >= int(early_cfg["patience_evaluations"])
        ):
            break

    if accelerator.is_main_process:
        summary = {
            "best": best,
            "wall_clock_seconds": time.perf_counter() - start_time,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated(accelerator.device) / (1024**2)
                if accelerator.device.type == "cuda"
                else None
            ),
            "note": "Test was not evaluated; load best_adapter only after freezing the configuration.",
        }
        write_json(output_dir / "training_summary.json", summary)


if __name__ == "__main__":
    main()
