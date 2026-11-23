# Reliability of image explanations under matched input changes

Saliency maps suggest which pixels influence a classifier’s predictions. Two nearly identical inputs can receive the same class label and different maps. Conversely, a constant map can agree with itself under every edit while giving the user no information about how the classifier uses the image. The project evaluates agreement under input changes, dependence on learned parameters, and changes in target probability when selected pixels are removed or restored.

The task is binary cat-versus-dog classification on Oxford-IIIT Pet. We evaluate Grad-CAM, Integrated Gradients and Occlusion using one ImageNet-pretrained ResNet18 checkpoint. Geometric edits change pixel coordinates, so comparison requires a common spatial frame and valid overlap. Photometric edits use RGB pixel units to avoid effects from changing the model’s normalized tensors.

## Research questions

1. **Does the stability ranking change when classic edits are replaced by VAE neighbours?** Compute TCERS on prediction-preserving pairs. For the primary comparison every pair starts at the same image decoded at the posterior mean. Match classic edits with latent edits within each image and seed using pixel RMSE. Compare paired mean TCERS changes and test whether those changes differ between explainers.
2. **Does stability correlate with faithfulness across images?** Use deletion and insertion AUC at the fixed reference class. Compute within-method Spearman correlations for each baseline using image-level means over common seeds. Include independent random attribution and the constant control to interpret agreement scores.
3. **Do model randomization results agree with the stability ordering?** Implement both Adebayo reset variants. Compare the Adebayo-style similarity curves with the published trends. Rank methods by the mean reduction in signed rank similarity across cascading reset stages and compare that ordering descriptively with the matched VAE ordering.

## Design decisions

- Use one fixed class-stratified test cohort and three independent evaluation seeds. Train one VAE per seed on the training subset. Keep the training-validation partition fixed. The classifier checkpoint is common across experiments.
- Train the VAE with mean pixel MSE and a weighted KL summed across latent dimensions and averaged across images. Select checkpoints by deterministic validation loss evaluated at the posterior mean. Evaluate reconstruction error and classifier label agreement on the same held-out cohort.
- Add isotropic Gaussian noise of configurable standard deviation to the latent mean. Decode and resize both members to the classifier resolution. Compare against classic edits applied to the decoded reference. Pair candidates without replacement using minimum total absolute RMSE gap within a prespecified absolute caliper. Only pairs that preserve the fixed decoded prediction qualify. Export distance distributions, match coverage and prediction retention.
- Export raw-image classic results as a secondary comparison. Align rotation and resize maps with the forward transform. Pool spatial metrics to a fixed measurement grid. Score only valid overlap after rotation. For the sanity check use the classifier-resolution maps, keeping the signed IG channel sum before its magnitude transform.
- Deletion and insertion progressively replace or restore the highest-ranked pixels. Use blur and per-image RGB mean baselines. Attribution magnitudes determine order. A shared seeded permutation resolves ties before sorting. For the constant control this induces a random pixel order independent of the model.
- Report mean, standard deviation, and percentile bootstrap 95% intervals across image means. Pair at the image level. Apply Holm within each reported test family. Report rank-biserial effect sizes and bootstrap intervals for mean differences. These intervals describe variation across test images conditional on the fitted classifier and the selected evaluation seeds.

The VAE captures a learned approximation of the image manifold. Reconstruction agreement and the size of the matched cohort determine how broadly the experiment can speak about the original data.
