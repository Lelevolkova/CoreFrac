# CoreFrac

Benchmark, code, and adaptation protocol for **drill-core fracture segmentation
by prompt evolution**: a frozen SAM 3 concept segmentor is retargeted to a
specialist industrial domain by evolving only its natural-language prompt, and
its masks are distilled into a compact on-premises U-Net.

Accompanies *Adapt the Words, Freeze the Model: LLM Prompt Evolution for
Domain-Shift-Robust Segmentation* (EMNLP 2026, Industry Track).

## What is here

```text
dataset/          200 full-resolution drill-core photographs with expert masks,
                  450 benchmark patches, and the source-grouped split
code/             evaluation, distillation, LoRA training, and zero-shot
                  baseline scripts, plus the prompt-evolution problem definition
configs/lora/     matched-cap rank-8 LoRA protocol on SAM 3
prompts/          seed prompts and the candidate prompt set
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

export COREFRAC_ROOT="$PWD/dataset/patches"
export COREFRAC_SPLIT_MANIFEST="$PWD/dataset/patches/splits_grouped/corefrac_grouped_split_manifest.json"
export COREFRAC_TRAIN_N_SAMPLES=96
```

`infer_unet.py` needs only a distilled U-Net checkpoint, which you produce with
`distill_unet.py` (see [Model weights](#model-weights)). Everything that scores
SAM 3 additionally needs a SAM 3 checkpoint, which is an external dependency
under its own license and is not redistributed here.

## Splits

Headline results use the **source-grouped** split: 99 / 179 / 172 patches from
18 / 25 / 23 drill cores with zero source-core overlap. Whole cores are
assigned to one split, so no crop of a core can appear in two splits.

| Split | Cores | Positive | Empty | Total |
|-------|------:|---------:|------:|------:|
| train |    18 |       83 |    16 |    99 |
| val   |    25 |      143 |    36 |   179 |
| test  |    23 |      134 |    38 |   172 |

Every adapted method is capped at the same 96-patch adaptation pool
(`splits_grouped/adaptation_pool_96.json`, 80 positive + 16 empty); the three
remaining training patches are excluded from every method and are not moved to
validation or test. `splits_grouped/build_grouped_split.py` rebuilds the split
deterministically and re-runs the integrity checks reported in
`splits_grouped/validation_report.md`.

`dataset/patches/splits.json` is the **superseded** patch-level split from the
first release. It is diversity-stratified but not source-aware: 61 of 66 cores
occur in more than one split. It is kept only so the earlier baseline numbers
remain reproducible; do not use it for supervised training.

## Reproducing the paper

| Paper item | Entry point |
|---|---|
| Frozen SAM 3 prompt scores (Table 1) | `code/eval_ladder_testset_soft.py`, then `code/aggregate_test_soft.py` |
| Zero-shot open-vocabulary baselines (Table 1) | `code/zero_shot_open_seg.py` |
| Distilled U-Net (Table 1) | `code/distill_unet.py --stage all` |
| Matched-cap rank-8 LoRA (Table 1) | `code/train_sam3_lora.py` with `configs/lora/corefrac_sam3_lora_r8_grouped.yaml` |
| Maintenance under domain shift (Table 2) | `code/eval_transfer.py` |
| Paired confidence intervals | `code/cluster_bootstrap.py`, fed a long-format CSV of per-patch, per-seed scores (header documented in the script) |
| Prompt evolution loop | `code/problems/prompts/corefrac/` (requires the GigaEvo framework) |

The primary metric is soft tolerance-F1 at a 2-pixel tolerance, with strict
pixel Dice reported alongside as a conservative reference. An empty prediction
on an empty patch scores 1.0, so positives-only numbers are primary and
all-patch numbers secondary. See `code/problems/prompts/corefrac/metrics.yaml`.

Uncertainty follows the paper: three independent seeds reported as mean and
sample standard deviation, and paired differences from a source-core cluster
bootstrap (10,000 replicates) rather than treating patches as independent.

## Prompts

`prompts/initial_programs_grouped/` holds the eight seed phrases the grouped
evolution starts from; `code/problems/prompts/corefrac/initial_programs/` holds
the seeds of the earlier ungrouped run. `prompts/prompt_pool.json` is the
candidate set used for the manual prompt comparison, with the design rationale
and expected failure mode recorded for each phrase, and `prompts/prompts.txt`
is the same set as a flat `ID | phrase` list, including the earlier champion
kept as an unchanged control across comparisons.

## Model weights

No trained weights are distributed here. The distilled U-Net (7.76 M
parameters, 31.1 MB) is reproduced with `code/distill_unet.py --stage all`, and
the matched-cap LoRA adapters with `code/train_sam3_lora.py` using
`configs/lora/corefrac_sam3_lora_r8_grouped.yaml`. The reported adapter has
1.29 M trainable parameters. SAM 3 weights are an external dependency under
their own license and are not redistributed.

## Licenses

Code is MIT (`code/LICENSE`); the dataset is CC BY 4.0 (`dataset/LICENSE`).
SAM 3 weights are an external dependency under their own license and are not
redistributed.

## Citation

See `CITATION.cff`.
