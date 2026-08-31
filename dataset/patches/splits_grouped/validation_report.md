# CoreFrac source-grouped split validation

Input: `dataset/patches/manifest.jsonl`
Input manifest SHA-256: `cef8ab4656a910f9594e6d871c0444f69c2954ac4cf2cc217e4551367baacbdc`

## Result

| split | positive | empty | total | source cores |
|---|---:|---:|---:|---:|
| train | 83 | 16 | 99 | 18 |
| val | 143 | 36 | 179 | 25 |
| test | 134 | 38 | 172 | 23 |

- Sample overlap across grouped splits: 0.
- Source-core overlap across grouped splits: 0.
- Original patch-level split: 61/66 source cores occur in more than one split.
- Fixed adaptation pool: 96 patches (80 positive, 16 empty).
- Grouped-train patches excluded by the 96-patch budget: 3: `0435_positive_core_0095__y02304_02560`, `0440_positive_core_0167__y01344_01600`, `0442_positive_core_0082__y00000_00256`.
- All 450 image/mask pairs exist; image and mask sizes match crop metadata.
- All masks are binary {0,255}; empty/positive labels agree with mask contents.

## Reproduction

The builder reproduces the released `_assign_grouped_splits` implementation: sort source-core groups, shuffle them with Python `random.Random(42)`, and assign each whole group to train, validation, then test until the corresponding patch target has been reached. Whole-group assignment changes the nominal 96/178 targets to realized 99/179 sizes; test receives the remaining 172 patches.

For the existing loader, point `COREFRAC_SPLIT_MANIFEST` to `corefrac_grouped_split_manifest.json` and keep `COREFRAC_TRAIN_N_SAMPLES=96`. The loader sorts each split and takes the first 96 training IDs, exactly matching `adaptation_pool_96.json`.
