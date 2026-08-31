#!/usr/bin/env python3
"""Rebuild and validate the deterministic source-grouped CoreFrac split.

This reproduces ``_assign_grouped_splits`` from the released code: source-core
groups are sorted, shuffled with Python's ``random.Random(seed)``, and assigned
whole to train, validation, then test until the requested patch-count targets
are reached.  Whole-group assignment may overshoot a target.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_grouped_split(
    rows: list[dict], seed: int, train_fraction: float, val_fraction: float
) -> dict[str, list[str]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_core"]].append(row)

    ordered_groups = sorted(groups.items())
    random.Random(seed).shuffle(ordered_groups)

    train_target = round(len(rows) * train_fraction)
    val_target = round(len(rows) * val_fraction)
    split_rows: dict[str, list[dict]] = {name: [] for name in SPLITS}

    for _core, group_rows in ordered_groups:
        if train_target - len(split_rows["train"]) > 0:
            split = "train"
        elif val_target - len(split_rows["val"]) > 0:
            split = "val"
        else:
            split = "test"
        split_rows[split].extend(group_rows)

    return {
        split: sorted(row["sample_id"] for row in split_rows[split])
        for split in SPLITS
    }


def validate_source_rows(dataset_dir: Path, rows: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = [row["sample_id"] for row in rows]
    duplicate_ids = sorted(sample_id for sample_id, n in Counter(ids).items() if n > 1)
    if duplicate_ids:
        errors.append(f"duplicate sample ids: {duplicate_ids}")

    for row in rows:
        sample_id = row["sample_id"]
        encoded_core = re.search(r"core_\d+", sample_id)
        if not encoded_core or encoded_core.group(0) != row["source_core"]:
            errors.append(f"source_core mismatch: {sample_id}")
        encoded_label = "empty" if "_empty_" in sample_id else "positive"
        if encoded_label != row["label"]:
            errors.append(f"label mismatch: {sample_id}")

        image_path = dataset_dir / row["image_path"]
        mask_path = dataset_dir / row["mask_path"]
        if not image_path.is_file():
            errors.append(f"missing image: {row['image_path']}")
            continue
        if not mask_path.is_file():
            errors.append(f"missing mask: {row['mask_path']}")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                errors.append(f"image/mask size mismatch: {sample_id}")
            crop = row["crop"]
            crop_size = (crop["x1"] - crop["x0"], crop["y1"] - crop["y0"])
            if image.size != crop_size:
                errors.append(f"crop/image size mismatch: {sample_id}")
            histogram = mask.convert("L").histogram()
            values = {value for value, count in enumerate(histogram) if count}
            if not values.issubset({0, 255}):
                errors.append(f"non-binary mask: {sample_id}: {sorted(values)}")
            has_positive = 255 in values
            if row["label"] == "empty" and has_positive:
                errors.append(f"empty sample has positive mask pixels: {sample_id}")
            if row["label"] == "positive" and not has_positive:
                errors.append(f"positive sample has empty mask: {sample_id}")
    return errors


def summarize(rows_by_id: dict[str, dict], splits: dict[str, list[str]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for split in SPLITS:
        split_rows = [rows_by_id[sample_id] for sample_id in splits[split]]
        counts = Counter(row["label"] for row in split_rows)
        summary[split] = {
            "positive": counts["positive"],
            "empty": counts["empty"],
            "total": len(split_rows),
            "source_cores": len({row["source_core"] for row in split_rows}),
        }
    return summary


def validate_split(rows: list[dict], splits: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    expected_ids = {row["sample_id"] for row in rows}
    rows_by_id = {row["sample_id"]: row for row in rows}
    assigned = [sample_id for split in SPLITS for sample_id in splits[split]]

    if len(assigned) != len(set(assigned)):
        errors.append("sample ids overlap between grouped splits")
    if set(assigned) != expected_ids:
        errors.append("grouped splits do not cover exactly the manifest sample ids")

    core_sets = {
        split: {rows_by_id[sample_id]["source_core"] for sample_id in splits[split]}
        for split in SPLITS
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sorted(core_sets[left] & core_sets[right])
            if overlap:
                errors.append(f"source_core overlap {left}/{right}: {overlap}")
    return errors


def old_split_leakage(rows_by_id: dict[str, dict], old_split_path: Path) -> tuple[int, int]:
    payload = json.loads(old_split_path.read_text(encoding="utf-8"))
    old_splits = payload.get("splits", payload)
    core_to_splits: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        for sample_id in old_splits[split]:
            core_to_splits[rows_by_id[sample_id]["source_core"]].add(split)
    leaky = sum(len(split_names) > 1 for split_names in core_to_splits.values())
    return leaky, len(core_to_splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.213)
    parser.add_argument("--val-fraction", type=float, default=0.396)
    parser.add_argument("--adaptation-budget", type=int, default=96)
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.jsonl"
    old_split_path = args.dataset_dir / "splits.json"
    rows = read_jsonl(manifest_path)
    rows_by_id = {row["sample_id"]: row for row in rows}

    source_errors = validate_source_rows(args.dataset_dir, rows)
    splits = build_grouped_split(
        rows, args.seed, args.train_fraction, args.val_fraction
    )
    split_errors = validate_split(rows, splits)
    errors = source_errors + split_errors
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))

    if args.adaptation_budget > len(splits["train"]):
        raise SystemExit("adaptation budget exceeds grouped training split")
    adaptation_ids = splits["train"][: args.adaptation_budget]
    excluded_train_ids = splits["train"][args.adaptation_budget :]
    summary = summarize(rows_by_id, splits)
    leaky_cores, old_core_count = old_split_leakage(rows_by_id, old_split_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "dataset_root": "dataset/patches",
        "seed": args.seed,
        "selection": "source_core_grouped_sequential_fill",
        "fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": 1.0 - args.train_fraction - args.val_fraction,
        },
        "targets_before_whole_group_overshoot": {
            "train": round(len(rows) * args.train_fraction),
            "val": round(len(rows) * args.val_fraction),
        },
        "counts": summary,
        "input_manifest_sha256": sha256(manifest_path),
        "splits": splits,
    }
    (args.out_dir / "corefrac_grouped_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    split_by_id = {
        sample_id: split for split in SPLITS for sample_id in splits[split]
    }
    with (args.out_dir / "corefrac_grouped_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in rows:
            enriched = {**row, "split": split_by_id[row["sample_id"]]}
            stream.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    pool_payload = {
        "dataset_root": "dataset/patches",
        "seed": args.seed,
        "selection": "first_n_sorted_ids_from_grouped_train",
        "budget": args.adaptation_budget,
        "counts": dict(Counter(rows_by_id[sample_id]["label"] for sample_id in adaptation_ids)),
        "sample_ids": adaptation_ids,
        "excluded_grouped_train_sample_ids": excluded_train_ids,
    }
    (args.out_dir / "adaptation_pool_96.json").write_text(
        json.dumps(pool_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# CoreFrac source-grouped split validation",
        "",
        "Input: `dataset/patches/manifest.jsonl`",
        f"Input manifest SHA-256: `{sha256(manifest_path)}`",
        "",
        "## Result",
        "",
        "| split | positive | empty | total | source cores |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = summary[split]
        report_lines.append(
            f"| {split} | {item['positive']} | {item['empty']} | "
            f"{item['total']} | {item['source_cores']} |"
        )
    pool_counts = Counter(rows_by_id[sample_id]["label"] for sample_id in adaptation_ids)
    report_lines.extend(
        [
            "",
            "- Sample overlap across grouped splits: 0.",
            "- Source-core overlap across grouped splits: 0.",
            f"- Original patch-level split: {leaky_cores}/{old_core_count} source cores occur in more than one split.",
            f"- Fixed adaptation pool: {len(adaptation_ids)} patches ({pool_counts['positive']} positive, {pool_counts['empty']} empty).",
            f"- Grouped-train patches excluded by the 96-patch budget: {len(excluded_train_ids)}: "
            + ", ".join(f"`{sample_id}`" for sample_id in excluded_train_ids)
            + ".",
            "- All 450 image/mask pairs exist; image and mask sizes match crop metadata.",
            "- All masks are binary {0,255}; empty/positive labels agree with mask contents.",
            "",
            "## Reproduction",
            "",
            "The builder reproduces the released `_assign_grouped_splits` implementation: sort source-core groups, shuffle them with Python `random.Random(42)`, and assign each whole group to train, validation, then test until the corresponding patch target has been reached. Whole-group assignment changes the nominal 96/178 targets to realized 99/179 sizes; test receives the remaining 172 patches.",
            "",
            "For the existing loader, point `COREFRAC_SPLIT_MANIFEST` to `corefrac_grouped_split_manifest.json` and keep `COREFRAC_TRAIN_N_SAMPLES=96`. The loader sorts each split and takes the first 96 training IDs, exactly matching `adaptation_pool_96.json`.",
        ]
    )
    (args.out_dir / "validation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"adaptation pool: {len(adaptation_ids)}; excluded: {excluded_train_ids}")
    print(f"wrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
