# CoreFrac: prompt evolution for drill-core fracture segmentation (code)

Anonymized code release accompanying the paper. It covers the full pipeline:
evolving SAM3 text prompts on the CoreFrac patch benchmark, scoring fixed-prompt
baselines, distilling the SAM3 teacher into a deployable UNet, and the
open-vocabulary zero-shot baselines.

All author-, institution-, and site-identifying information has been removed.
Drill-core identifiers in the dataset are anonymized to `core_0001 … core_0200`
(see the separate dataset release).

## Layout

```text
code/
  requirements.txt
  distill_unet.py            # SAM3 pseudo-labels -> UNet knowledge distillation
  infer_unet.py              # run the distilled UNet to produce binary masks
  eval_ladder_testset_soft.py# soft tolerance-F1 ladder eval on the held-out test set
  aggregate_test_soft.py     # aggregate shard results into a summary
  eval_transfer.py           # cross-domain (maintenance) transfer evaluation
  zero_shot_open_seg.py      # CLIPSeg / OWLv2+SAM / Florence-2 zero-shot baselines
  problems/prompts/corefrac/
    config.py                # all paths/params via env vars (no hardcoded paths)
    task_description.txt      # task spec handed to the evolutionary LLM
    metrics.yaml             # fitness / reporting metric definitions
    corefrac_split_manifest.json  # deterministic train/val/test split
    initial_programs/        # seed prompts the evolution starts from
    utils/                   # dataset loader, SAM3 tool, metrics, validation
    scripts/                 # run_evolution.sh, baseline + manifest helpers
    test.py / validate.py    # evolution harness + fitness evaluation
    README.md                # problem-specific details
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point the code at the released dataset:
export COREFRAC_ROOT=/path/to/dataset/patches
export COREFRAC_SPLIT_MANIFEST=problems/prompts/corefrac/corefrac_split_manifest.json
```

All inputs are configured through environment variables (`COREFRAC_ROOT`,
`SAM3_REPO`, `SAM3_MODEL_NAME`, `INF_URL`, …). There are no machine-specific
absolute paths in the code.

## What runs standalone

These scripts run directly against the released dataset (a SAM3 checkpoint is
required for the SAM3-based scoring scripts; `infer_unet.py` only needs the UNet
weights):

- `distill_unet.py`, `infer_unet.py`
- `eval_ladder_testset_soft.py`, `aggregate_test_soft.py`, `eval_transfer.py`
- `zero_shot_open_seg.py`
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
