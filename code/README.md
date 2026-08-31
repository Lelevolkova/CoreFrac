# CoreFrac: prompt evolution for drill-core fracture segmentation (code)

Code accompanying the paper. It covers the full pipeline: evolving SAM3 text
prompts on the CoreFrac patch benchmark, scoring fixed-prompt baselines,
training the matched-cap rank-8 LoRA baseline, distilling the SAM3 teacher into
a deployable UNet, the open-vocabulary zero-shot baselines, and the paired
uncertainty analysis.

Drill-core identifiers in the dataset are anonymized to `core_0001 … core_0200`
and all site, well, operator, and depth metadata has been removed (see the
separate dataset release). Start from the repository-root `README.md` for the
split definitions and the map from paper results to entry points.

## Layout

```text
code/
  requirements.txt
  distill_unet.py            # SAM3 pseudo-labels -> UNet knowledge distillation
  infer_unet.py              # run the distilled UNet to produce binary masks
  train_sam3_lora.py         # matched-cap rank-8 LoRA on SAM 3 (Table 1)
  eval_ladder_testset_soft.py# soft tolerance-F1 ladder eval on the held-out test set
  aggregate_test_soft.py     # aggregate shard results into a summary
  eval_transfer.py           # cross-domain (maintenance) transfer evaluation
  zero_shot_open_seg.py      # CLIPSeg / OWLv2+SAM / Florence-2 zero-shot baselines
  cluster_bootstrap.py       # paired source-core cluster bootstrap for CIs
  problems/prompts/corefrac/
    config.py                # all paths/params via env vars (no hardcoded paths)
    task_description.txt      # task spec handed to the evolutionary LLM
    metrics.yaml             # fitness / reporting metric definitions
    corefrac_grouped_split_manifest.json  # source-grouped split (headline)
    corefrac_split_manifest.json          # superseded patch-level split
    initial_programs/        # seed prompts of the earlier ungrouped run
    utils/                   # dataset loader, SAM3 tool, metrics, validation
    scripts/                 # run_evolution.sh, baseline + manifest helpers
    test.py / validate.py    # evolution harness + fitness evaluation
    README.md                # problem-specific details
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point the code at the released dataset and the source-grouped split:
export COREFRAC_ROOT=/path/to/dataset/patches
export COREFRAC_SPLIT_MANIFEST=problems/prompts/corefrac/corefrac_grouped_split_manifest.json
export COREFRAC_TRAIN_N_SAMPLES=96
```

All inputs are configured through environment variables (`COREFRAC_ROOT`,
`SAM3_REPO`, `SAM3_MODEL_NAME`, `INF_URL`, …). There are no machine-specific
absolute paths in the code.

## What runs standalone

These scripts run directly against the released dataset (a SAM3 checkpoint is
required for the SAM3-based scoring scripts; `infer_unet.py` only needs the UNet
weights):

- `distill_unet.py`, `infer_unet.py`
- `train_sam3_lora.py` (PEFT / Accelerate; launch with
  `accelerate launch --num_processes 2` to match `num_gpus: 2` in the YAML. On a
  different GPU count, adjust `num_gpus` and `gradient_accumulation_steps`
  together so `expected_effective_batch_size: 8` still holds; `--dry-run`
  validates data, split integrity, and the adapter without training.)
- `eval_ladder_testset_soft.py`, `aggregate_test_soft.py`, `eval_transfer.py`
- `zero_shot_open_seg.py`
- `cluster_bootstrap.py` (needs only per-patch scores, no model)
- `problems/.../scripts/run_baseline_shard.py`, `aggregate_results.py`

## What requires the evolutionary framework

The prompt-evolution loop itself — `scripts/run_evolution.sh` and the
`test.py` harness (`problems.prompts.utils`) — runs inside the authors'
**GigaEvo** evolutionary framework (MAP-Elites + LLM mutation) and additionally
requires a SAM3 checkpoint, an OpenAI-compatible LLM inference endpoint, and a
Redis instance. The problem package here is the exact, readable problem
definition that framework drives; `config.py`, `task_description.txt`,
`initial_programs/`, and `utils/` fully specify the evolved task and fitness.

## Metrics

Primary metric is **soft tolerance-F1** (`tol=2px`); strict pixel Dice and IoU
are reported alongside it. See `metrics.yaml` and the problem `README.md`.

Headline comparisons run three independent seeds and report mean and sample
standard deviation. Paired differences between systems use a source-core
cluster bootstrap (`cluster_bootstrap.py`, 10,000 replicates over the 23 test
cores) rather than treating the 172 test patches as independent.
