# Project 4 results

Full protocol. Classifier test accuracy 97.7% on 3669 images.

## RQ1 · Classic and VAE stability

The matcher retained 809 pairs across 287 of 300 images. Classic RMSE 0.064 [0.061, 0.068] (n=287) · VAE RMSE 0.064 [0.061, 0.068] (n=287). Classifier agreement between original and reconstructed images was 0.300 [0.250, 0.350] (n=300).

| Method | Classic on raw | Classic on decoded, matched | VAE, matched |
| --- | --- | --- | --- |
| Grad-CAM | 0.987 [0.986, 0.989] (n=300) | 0.951 [0.946, 0.957] (n=287) | 0.988 [0.986, 0.989] (n=287) |
| IG | 0.954 [0.953, 0.956] (n=300) | 0.836 [0.829, 0.843] (n=287) | 0.872 [0.866, 0.878] (n=287) |
| Occlusion | 0.925 [0.922, 0.928] (n=300) | 0.787 [0.776, 0.798] (n=287) | 0.869 [0.861, 0.877] (n=287) |
| Random | 0.645 [0.645, 0.645] (n=300) | 0.645 [0.644, 0.645] (n=287) | 0.645 [0.645, 0.646] (n=287) |
| Constant | 1.000 [1.000, 1.000] (n=300) | 1.000 [1.000, 1.000] (n=287) | 1.000 [1.000, 1.000] (n=287) |

Matched classic order · Grad-CAM → IG → Occlusion.

Matched VAE order · Grad-CAM → IG → Occlusion.

The point-estimate ranking is the same in both matched arms. Paired differences below subtract matched classic TCERS from matched VAE TCERS. Image-level intervals and Holm-adjusted tests quantify uncertainty.

| Method | n | Mean Δ TCERS | 95% CI | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- | --- |
| Grad-CAM | 287 | 0.036 | [0.031, 0.042] | 0.919 | <0.001 |
| IG | 287 | 0.036 | [0.028, 0.044] | 0.545 | <0.001 |
| Occlusion | 287 | 0.082 | [0.071, 0.093] | 0.812 | <0.001 |

### VAE validation at the selected epoch

| Seed | Epoch | Validation MSE | Validation KL |
| --- | --- | --- | --- |
| 42 | 20 | 0.029 | 8.645 |
| 43 | 9 | 0.029 | 7.528 |
| 44 | 18 | 0.028 | 8.844 |

## RQ2 · Stability and faithfulness

The following table uses the `blur` baseline and decoded images. The CSVs include raw images and every configured baseline. A constant explanation induces a seeded random order under faithfulness evaluation. Its TCERS and AUC estimate different properties.

| Method | Decoded deletion ↓ | Decoded insertion ↑ |
| --- | --- | --- |
| Grad-CAM | 0.905 [0.902, 0.909] (n=300) | 0.913 [0.909, 0.916] (n=300) |
| IG | 0.913 [0.910, 0.916] (n=300) | 0.916 [0.913, 0.920] (n=300) |
| Occlusion | 0.909 [0.905, 0.913] (n=300) | 0.910 [0.906, 0.914] (n=300) |
| Random | 0.916 [0.913, 0.919] (n=300) | 0.916 [0.913, 0.920] (n=300) |
| Constant | 0.916 [0.913, 0.919] (n=300) | 0.916 [0.913, 0.919] (n=300) |

Spearman correlations for matched VAE TCERS and faithfulness are shown below. AUC directions differ, with lower deletion and higher insertion indicating better faithfulness. Correlations use image means from matching seeds.

| Method | Outcome | n | ρ [95% CI] | Holm p |
| --- | --- | --- | --- | --- |
| Constant | deletion_auc | 287 | — [—, —] | — |
| Constant | insertion_auc | 287 | — [—, —] | — |
| Grad-CAM | deletion_auc | 287 | 0.271 [0.153, 0.386] | <0.001 |
| Grad-CAM | insertion_auc | 287 | 0.265 [0.142, 0.385] | <0.001 |
| IG | deletion_auc | 287 | 0.404 [0.302, 0.506] | <0.001 |
| IG | insertion_auc | 287 | 0.419 [0.317, 0.517] | <0.001 |
| Occlusion | deletion_auc | 287 | 0.491 [0.400, 0.576] | <0.001 |
| Occlusion | insertion_auc | 287 | 0.501 [0.400, 0.585] | <0.001 |
| Random | deletion_auc | 287 | -0.012 [-0.134, 0.109] | 1.000 |
| Random | insertion_auc | 287 | -0.012 [-0.128, 0.102] | 1.000 |

## RQ3 · Adebayo parameter randomization

The cascading estimates below focus on the first and last ResNet18 resets. See the [layerwise curves](figures/sanity_checks.png) for both variants and [the published comparison protocol](../../docs/sanity_protocol.md) for architecture and metric differences.

| Method | FC signed ρ | Stem signed ρ | Stem magnitude ρ | Stem SSIM | Stem HOG |
| --- | --- | --- | --- | --- | --- |
| Grad-CAM | -0.012 [-0.050, 0.026] (n=300) | -0.052 [-0.068, -0.036] (n=300) | -0.052 [-0.068, -0.036] (n=300) | 0.261 [0.254, 0.266] (n=300) | 0.063 [0.060, 0.067] (n=300) |
| IG | -0.039 [-0.094, 0.015] (n=300) | -0.000 [-0.000, 0.000] (n=300) | 0.270 [0.253, 0.288] (n=300) | 0.292 [0.280, 0.304] (n=300) | 0.353 [0.336, 0.371] (n=300) |
| Occlusion | -0.169 [-0.191, -0.148] (n=300) | -0.006 [-0.024, 0.012] (n=300) | 0.025 [0.004, 0.046] (n=300) | 0.181 [0.168, 0.195] (n=300) | 0.913 [0.899, 0.924] (n=300) |
| Random | -0.000 [-0.000, 0.000] (n=300) | 0.000 [-0.000, 0.000] (n=300) | -0.000 [-0.000, 0.000] (n=300) | 0.032 [0.032, 0.033] (n=300) | 0.030 [0.028, 0.031] (n=300) |
| Constant | 1.000 [1.000, 1.000] (n=300) | 1.000 [1.000, 1.000] (n=300) | 1.000 [1.000, 1.000] (n=300) | 1.000 [1.000, 1.000] (n=300) | 1.000 [1.000, 1.000] (n=300) |

Adebayo et al. found that plain Grad-CAM changed with parameter resets. Signed IG rank similarity dropped after upper-layer resets while magnitude retained more structure. Sliding-window Occlusion extends their quantitative comparison.

Cascading parameter sensitivity (mean of 1 − signed Spearman across reset stages) · Occlusion → Grad-CAM → IG.

Matched VAE stability · Grad-CAM → IG → Occlusion.

The two point-estimate rankings differ. The ordering is descriptive for the three explanation methods. Adebayo et al. also reported an approximately zero signed and magnitude Spearman correlation between IG maps and Gaussian random masks. Here random and constant controls appear alongside the methods for every layer and metric. [Adebayo et al., Figures 2–4 and Appendix B](https://arxiv.org/html/1810.03292)
