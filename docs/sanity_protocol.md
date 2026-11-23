# Model parameter randomization reproduction

## Published protocol

Adebayo et al. compare an explanation from the trained model with one from a model with reset weights. The **cascading** variant resets successive stages from the logits toward the input, retaining earlier resets at each step. The **independent** variant starts from the trained model at each step and resets only the selected layer or block. For Inception v3, the authors reset filters, biases and batch normalization variables by block. The paper reports truncated-normal initialization and verifies the qualitative findings with a uniform distribution. [Sanity Checks, §3 and Appendix B](https://arxiv.org/html/1810.03292)

The paper reports Spearman rank correlation on signed maps and on the absolute values of the maps, plus SSIM and the Pearson correlation between HOG descriptors on signed ImageNet maps. Appendix B describes a five-pixel SSIM window and HOG cells of 16 × 16 pixels. Applying the absolute value **before** ranking measures persistence of attribution magnitude, including when signs change. [Sanity Checks, §2.2 and Appendix B](https://arxiv.org/html/1810.03292)

## Published results and comparison criteria

| Method | Published finding | Comparison on Oxford-IIIT Pet |
| --- | --- | --- |
| Grad-CAM | Plain Grad-CAM maps changed with parameter randomization. Guided Grad-CAM, which multiplies with Guided Backpropagation, retained more visual structure. | Report the change in layerwise similarity for plain Grad-CAM in both variants, with image-level confidence intervals. |
| Integrated Gradients | On Inception v3 and ImageNet, signed rank correlation fell toward zero soon after randomizing upper layers. Magnitude-based rank correlation, SSIM and HOG reflected more retained image structure. Independent examples also retained visible structure. | Compare signed and magnitude rank curves at early and late reset depths and assess whether the qualitative pattern recurs. |
| Sliding-window Occlusion | The main quantitative plots do not include sliding-window Occlusion. Appendix C Figure 36 shows an AlexNet example labeled “Perturbation Method.” | Report the measured Occlusion curves as an extension. |

Sources for the published findings are §§3.1–3.2, Figures 2–4 and Appendix C. The authors’ erratum clarifies that Guided Backpropagation can change when layers near the input are reset, despite low sensitivity to resets near the output. [Revised paper](https://arxiv.org/html/1810.03292) · [Authors’ erratum](https://github.com/adebayoj/sanity_checks_saliency#guided-backprop-errata)

As a scale reference, Appendix B reports signed Spearman of approximately `0.00084` and magnitude Spearman of `0.00059` between Integrated Gradients maps and Gaussian random masks for the 50-image ImageNet calibration exercise. These are published calibration values. The project’s random and constant controls are evaluated on the Oxford-IIIT Pet setup. [Sanity Checks, Appendix B](https://arxiv.org/html/1810.03292#A2)

ResNet18 has different block boundaries, residual paths and activation maps from Inception v3. Oxford-IIIT Pet offers two classifier output classes and a narrower image domain than ImageNet. Image size, attribution preprocessing, and the choice of fixed explanation target also affect similarity. Comparisons therefore focus on the direction and depth of changes and on the signed-versus-magnitude difference for IG. Parameter sensitivity and input stability probe different interventions, so agreement with the stability ranking is descriptive. [Binder et al., §4](https://openaccess.thecvf.com/content/CVPR2023/papers/Binder_Shortcomings_of_Top-Down_Randomization-Based_Sanity_Checks_for_Evaluations_of_Deep_CVPR_2023_paper.pdf)
