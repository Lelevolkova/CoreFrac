"""Aggregate CoreFrac baseline shard JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "fitness": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "success_rate": 0.0,
            "avg_latency_sec": 0.0,
            "by_split": {},
            "by_prompt": {},
        }

    def one(group: list[dict[str, Any]]) -> dict[str, float]:
        n = len(group)
        return {
            "n": n,
            "fitness": sum(float(r["dice"]) for r in group) / n,
            "iou": sum(float(r["iou"]) for r in group) / n,
            "precision": sum(float(r["precision"]) for r in group) / n,
            "recall": sum(float(r["recall"]) for r in group) / n,
            "success_rate": sum(1.0 if r["success"] else 0.0 for r in group) / n,
            "avg_latency_sec": sum(float(r["latency_sec"]) for r in group) / n,
        }

    summary = one(rows)
    summary["by_split"] = {
        split: one([r for r in rows if str(r["split"]) == split])
        for split in sorted({str(r["split"]) for r in rows})
    }
    summary["by_prompt"] = {
        prompt: one([r for r in rows if str(r["prompt_name"]) == prompt])
        for prompt in sorted({str(r["prompt_name"]) for r in rows})
    }
    summary["n_errors"] = sum(1 for r in rows if r.get("error"))
    summary["n_success"] = sum(1 for r in rows if r.get("success"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="shard_*.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob(args.pattern))
    rows = load_rows(paths)
    rows.sort(key=lambda r: (str(r["prompt_name"]), str(r["split"]), str(r["sample_id"])))

    summary = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
