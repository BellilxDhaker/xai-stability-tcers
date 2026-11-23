# Project 4 implementation plan

| Grading criterion | Work |
| --- | --- |
| 1. Problem and RQs (5 pts) | Define stability, faithfulness, and model sensitivity. Specify the samples, comparisons, and outcomes for RQ1–RQ3. |
| 2. Critical literature review (7 pts) | Compare recent papers and connect their assumptions and limitations to the matched VAE neighbourhood experiment. Index publisher copies and foundational papers. |
| 3. Published method (5 pts) | Reproduce Adebayo et al.'s cascading and independent parameter randomization with fixed targets, signed and absolute Spearman, SSIM, HOG, and bootstrap intervals. Compare the measured trends with the paper. |
| 4. Personal contribution (3 pts) | Train a convolutional VAE on the training subset and compare latent neighbours with classic perturbations at matched pixel change. Report retention, reconstruction, uncertainty, effect sizes, Holm-adjusted tests, and the links to faithfulness and sanity checks. |

1. Preserve the official splits and checkpoint format. Move ImageNet normalization into the classifier and fix geometric alignment. Test identity, metric ranges, and degenerate TCERS cases.
2. Add common configuration, stratified evaluation, batched explainers, and map caching. Generate deletion and insertion curves with random and constant controls.
3. Implement both sanity checks, then train the VAE and evaluate classic and decoded perturbations on the same test cohort and classifier. Rerun inference and statistics with three evaluation seeds.
4. Generate CSV summaries, figures, notebooks, and a five-slide Marp deck. Verify the test suite and quick workflow at each stage, and record full-run status in CHECKLIST.md.
