# CoreFrac dataset

Geological drill-core fracture-segmentation dataset accompanying the paper.
Images are daylight photographs of single drill-core columns laid out from
core boxes, following standard geological core-photography practice. All
site/well/operator/depth metadata has been removed; cores are anonymized to
`core_0001 … core_0200`, and every file was re-encoded to strip EXIF/metadata.

## Contents

```text
dataset/
  full_res/
    images/   core_XXXX.png         # 200 full-resolution core columns (RGB)
    masks/    core_XXXX_mask.png     # binary fracture masks {0,255}
  patches/
    images/   <id>.png              # 450 benchmark patches (RGB)
    masks/    <id>_mask.png         # binary fracture masks {0,255}
    manifest.jsonl                  # per-patch metadata (id, label, source_core, crop, features)
    splits_grouped/                 # source-grouped split (use this one)
    splits.json                     # superseded patch-level split
```

## Patch benchmark

450 patches: 360 positive (contain fractures) + 90 empty. Patch ids encode
`<index>_<label>_<source_core>__y<y0>_<y1>` and each patch records its source
core and crop box in `manifest.jsonl`.

### Source-grouped split (headline)

`splits_grouped/corefrac_grouped_split_manifest.json` assigns **whole source
cores** to a split, so no two crops of the same core land in different splits.

| Split | cores | positive | empty | total |
|-------|------:|---------:|------:|------:|
| train |    18 |       83 |    16 |    99 |
| val   |    25 |      143 |    36 |   179 |
| test  |    23 |      134 |    38 |   172 |

Adapted methods are capped at a fixed 96-patch pool (80 positive + 16 empty),
listed in `splits_grouped/adaptation_pool_96.json`; the three remaining
training patches are excluded from every method and are not moved to another
split. `splits_grouped/build_grouped_split.py` rebuilds the assignment
deterministically (groups sorted, shuffled with `random.Random(42)`, filled
whole into train, then val, then test) and re-runs the integrity checks
recorded in `splits_grouped/validation_report.md`: zero sample overlap, zero
source-core overlap, zero spatial overlap, and mask/label agreement across all
450 pairs.

### Superseded patch-level split

`splits.json` is the diversity-stratified split of the first release
(farthest-point sampling over per-patch features, seed 42; 96/178/176 patches).
It spans the feature space but is *not* grouped by source: 61 of 66 source
columns touch more than one split. It is kept so the earlier baseline numbers
stay reproducible. Do not use it for supervised training.

Masks are 8-bit single-channel PNGs with values in `{0, 255}` (255 = fracture).
The primary evaluation metric in the paper is soft tolerance-F1 (`tol=2px`);
strict pixel Dice and IoU are reported alongside it.

Released under CC BY 4.0; see `LICENSE`.
