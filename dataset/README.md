# CoreFrac dataset (anonymized)

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
    splits.json                     # train / val / test sample ids (seed 42)
```

## Patch benchmark

450 patches: 360 positive (contain fractures) + 90 empty. Patch ids encode
`<index>_<label>_<source_core>__y<y0>_<y1>` and each patch records its source
core and crop box in `manifest.jsonl`. Splits are **diversity-stratified at the
patch level** (farthest-point sampling over per-patch features, seed 42): they span
the feature space but are *not* grouped by source, so crops from the same source
column can appear in different splits (61 of 66 source columns touch more than one
split). This is fine for prompt evolution (the evolved genome is a single global
text phrase) but means the patch split is not leakage-free for per-patch supervised
training; group by `source_core` if you need a source-aware split.

| Split | positive | empty | total |
|-------|---------:|------:|------:|
| train |       77 |    19 |    96 |
| val   |      142 |    36 |   178 |
| test  |      141 |    35 |   176 |

Masks are 8-bit single-channel PNGs with values in `{0, 255}` (255 = fracture).
The primary evaluation metric in the paper is soft tolerance-F1 (`tol=2px`);
strict pixel Dice and IoU are reported alongside it.

See `LICENSE` for usage terms during the review period.
