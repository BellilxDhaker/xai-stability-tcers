# Dataset: Oxford-IIIT Pet

- **Source:** Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. *CVPR*.
- **Homepage:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Content:** 7,349 images of 37 cat and dog breeds (≈ 200 per breed), with breed, species and pixel-level trimap annotations.
- **Licence:** Creative Commons Attribution-ShareAlike 4.0 International; image copyright remains with the original owners.
- **Size:** about 790 MB of images plus 19 MB of annotations (not committed to this repository).
- **Why this dataset:** natural photos with a clear object of interest, official train/test splits, and a binary cat/dog task on which a fine-tuned ResNet18 is highly accurate, so explanation instability is not caused by a weak classifier. Its head bounding boxes and trimaps also allow a future localisation check of the explanations (e.g. pointing game).

## How it is obtained

The dataset is downloaded automatically on first run by
`torchvision.datasets.OxfordIIITPet` (see `src/data/oxford_pets.py`) into
`ExperimentConfig.data_root` (default `data/`, git-ignored). No manual step is
needed.

## How it is used in this project

| Setting | Value |
|---|---|
| Target | `binary-category`: 0 = cat, 1 = dog |
| Splits | Official `trainval` and `test`; validation = 15 % of `trainval`, seed 42 |
| Sizes | train 3,128 · validation 552 · test 3,669 |
| Class balance | ≈ 32 % cat / 68 % dog in every split |
| Preprocessing | Resize to 224×224, ImageNet mean/std normalisation |
| Training augmentation | Random horizontal flip, random rotation ±10° (training only) |
| Stability study | First 8 images of the official test split (paired across methods) |

Diagnostics: `experiments/results/dataset_split_summary.csv`,
`dataset_class_balance.csv`, `dataset_image_geometry.csv`,
`figures/dataset_samples.png`, `figures/dataset_aspect_ratio.png`.
