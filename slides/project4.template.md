---
marp: true
size: 16:9
paginate: true
style: |
  section { background: #f6f8fb; color: #131f35; font-family: Arial, sans-serif; padding: 44px 66px; font-size: 25px; justify-content: flex-start; }
  section::after { color: #526683; font-size: 16px; }
  h1 { color: #0a5270; font-size: 49px; margin: 0 0 18px; }
  h2 { color: #0a5270; font-size: 38px; margin: 0 0 15px; }
  h3 { color: #0a5270; font-size: 24px; margin: 12px 0 4px; }
  p { margin: 8px 0 16px; }
  table { font-size: 17px; width: 100%; margin: 8px 0 12px; }
  th { background: #e0edf3; }
  blockquote { color: #42546b; font-size: 19px; }
  strong { color: #0a5270; }
  section.contribution p { margin: 5px 0 9px; }
  section.contribution td, section.contribution th { padding: 0.3em 0.65em; }
---

# When are pet classifier explanations reliable?

**Master 2 · RNA et Deep Learning · Project 4**

Grad-CAM, Integrated Gradients and Occlusion explain the predicted class of a ResNet18 trained on Oxford-IIIT Pet. Agreement under image edits, dependence on trained weights, and effects on the class score measure different properties.

1. Does the stability ranking change between matched classic and VAE edits?
2. Does an image's stability correlate with deletion and insertion AUC?
3. Does the layer-randomization sensitivity ranking agree with the stability ranking?

> @@MODE@@ · @@CLASSIFIER@@

---

## What existing evaluations establish

| Study | Contribution | Scope and limitation |
| --- | --- | --- |
| OpenXAI · Agarwal et al., 2022 | Benchmarks stability, faithfulness and explanation fairness. | Tabular evaluation and synthetic ground truth limit transfer to image attributions. |
| Binder et al., 2023 | Diagnoses progressively randomized networks and their attribution maps. | Layer order and normalization can influence sensitivity scores. |
| DiME · Jeanneret et al., 2022 | Uses diffusion to produce image counterfactuals. | Class-flipping edits differ from local, class-preserving neighbourhoods. |

The experiment compares classic and VAE edits around a common decoded image, at matched pixel RMSE. Faithfulness and parameter sensitivity are measured directly on the same test cohort.

---

## Reproducing Adebayo’s sanity checks

Reset ResNet18 from logits to stem, both cumulatively and one block at a time. Hold the original target class fixed. Compare signed and magnitude rank, SSIM, and HOG.

@@SANITY_VISUAL@@

> In the paper, plain Grad-CAM changed and signed IG lost agreement sooner than IG magnitude. Sliding-window Occlusion extends the comparison.

---

## Stability with VAE neighbours

<!-- _class: contribution -->

**Encode → decode → perturb latent mean → decode.** Match classic and VAE edits by pixel RMSE when predictions agree. TCERS averages SSIM, Spearman and cosine scaled to [0, 1].

@@MATCH@@

@@CONTRIBUTION@@

@@VAE_VISUAL@@

---

## What the comparisons can establish

- **RQ1** · @@RANK@@
- **RQ2** · @@RQ2@@ Lower deletion AUC and higher insertion AUC indicate better faithfulness.
- **RQ3** · Cascading parameter sensitivity order · @@SENSITIVITY@@.

Original and reconstructed predictions agreed at a rate of @@AGREEMENT@@. Matched edits preserve the decoded image's prediction. Pixel RMSE controls average change.

> Next experiments · vary the VAE capacity and test whether the ordering transfers to another classifier or dataset.
