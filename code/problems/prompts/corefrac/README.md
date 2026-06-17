# CoreFrac prompt-evolution problem

Prompt-evolution problem for detecting fractures in geological drill-core
images with a frozen CARL/SAM3 chain. The evolved genome is a single SAM3
text prompt returned by `entrypoint()`.

## Dataset

Default root: `data/corefrac/patches` (override with `COREFRAC_ROOT`).
The loader accepts either of these mask layouts:

```text
<root>/
  images/                # core_XXXX patches (.png/.jpg)
  masks/                 # <stem>_mask.png         (released layout)
  # ── or ──
  binary_masks/          # <stem>_binary_mask.png  (internal layout)
  manifest.jsonl         # optional, per-patch metadata
```

Images are paired to masks by normalized stem (the `_original`,
`_binary_mask`, and `_mask` suffixes are stripped before matching). Masks are
loaded as grayscale and thresholded with `COREFRAC_MASK_THRESHOLD`
(default `127`). Train/val/test splits are read from the split manifest
(`corefrac_split_manifest.json`) and are deterministic. The released manifest is
diversity-stratified at the patch level (farthest-point sampling, seed 42), so
crops from the same source column can appear in different splits; it is not
source-grouped. A source-aware splitter (`_assign_grouped_splits` in
`utils/dataset.py`, used when no manifest is supplied) groups by `source_core`.

## Evaluation modes

- `COREFRAC_EVAL_MODE=full`: run SAM3 on the full image.
- `COREFRAC_EVAL_MODE=patch_stitch`: run SAM3 on vertical windows, stitch
  predictions back into image coordinates, then score full-image metrics.

The primary metric is soft tolerance-F1 (`tol=2px`); strict pixel Dice and IoU
are reported alongside it.

## Baselines

Score a fixed prompt over a split with a single sharded worker:

```bash
python problems/prompts/corefrac/scripts/run_baseline_shard.py \
  --split test --shard 0 --num-shards 4 --out tmp_baseline
python problems/prompts/corefrac/scripts/aggregate_results.py \
  tmp_baseline
```

## Evolution

```bash
COREFRAC_EVAL_MODE=full bash \
  problems/prompts/corefrac/scripts/run_evolution.sh
```

`run_evolution.sh` reads the inference endpoint from `INF_URL` /
`LOCAL_INF_KEY` and the dataset from `COREFRAC_ROOT`; see the script header for
the full list of environment variables.
