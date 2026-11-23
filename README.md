# Explanation reliability on Oxford-IIIT Pet

Project 4 for the Master 2 RNA et Deep Learning GenAI mini-project. This study compares Grad-CAM, Integrated Gradients and sliding-window Occlusion for an ImageNet-pretrained ResNet18 that distinguishes cats from dogs. A convolutional VAE supplies local, class-preserving alternatives to classic input edits.

1. Does explanation stability rank the methods differently under matched classic and VAE perturbations?
2. Does per-image stability correlate with deletion and insertion faithfulness?
3. Does the ordering by sensitivity to randomized parameters agree with the stability ordering?

Read the [problem statement](docs/problem_statement.md), [state of the art](docs/state_of_the_art.md), [published comparison protocol](docs/sanity_protocol.md), [dataset instructions](data/README.md), and [paper downloads](papers/README.md).

## Run on CPU or Colab

Use Python 3.11–3.13. On a Colab T4, select a GPU runtime and substitute the `https://download.pytorch.org/whl/cu126` wheel index in the first command.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m src.train_vae
python -m src.run_sanity_checks
python -m src.run_stability
python -m src.run_faithfulness
python -m src.run_stats
```

Append `--quick` to **each** of the six commands for a small end-to-end smoke run. The dataset downloads on first use in both modes. Quick mode uses a small training subset, smaller tensors, one seed, reduced attribution steps and a classifier trained from scratch. It writes to `experiments/project4-quick/` and `checkpoints/quick/`. Its outputs are suitable for checking the workflow.

Full results go to `experiments/project4/` and checkpoints to `checkpoints/`. The target for the full pipeline is about two hours on a Colab T4. CPU runtime depends on the hardware and cached maps. Each stage writes its effective configuration. The [notebooks](notebooks/) call the same stages and display their outputs.

Use the same `--config overrides.json` with each stage to override any field in [ExperimentConfig](src/config.py). Common CLI overrides include `--test-samples`, `--data-root`, `--output-root`, `--checkpoint-root` and `--resnet-checkpoint`. For an existing compatible classifier checkpoint, pass `--resnet-checkpoint` to evaluate it and reuse it throughout. On a low-memory device reduce `forward_batch_size` and `evaluation_batch_size`. Keep the same batch configuration across comparison arms and reruns. The on-disk map cache is keyed by input pixels, model, targets and attribution settings. Set `cache_maps` to false when disk space is limited.

## Experimental protocol

- The classifier and VAEs use the official training data, with a fixed subset held out for validation. All experiments share a stratified cohort of at least 300 test images. The same classifier explains its original predicted target throughout each intervention.
- Apply edits to RGB pixels in `[0,1]`. Normalize inside the classifier. Forward-warp the reference map for geometry and score rotation only on valid overlap, with SSIM windows fully supported.
- TCERS averages SSIM, Spearman and cosine over prediction-preserving pairs. The first two use `(score + 1) / 2`; cosine is computed on nonnegative normalized maps. Identical constant maps score one under the explicit degenerate-map convention. Constant maps are ranked with seeded tie-breaking for deletion and insertion.
- Fit one VAE per evaluation seed. Compare reconstructions at the posterior mean with decoded Gaussian neighbours. Match predictively stable classic edits on the same reconstructions to predictively stable latent edits by pixel RMSE, within a fixed caliper and without replacement. Raw-image classic results give context across the complete cohort.
- Summaries report image-level sample counts, means, standard deviations and percentile bootstrap 95% CIs. Paired Wilcoxon tests use Holm correction and include rank-biserial effect sizes. Within-image aggregation accounts for repeated variants and evaluation seeds. See [protocol details](docs/problem_statement.md).

## Results

<!-- project4-results:start -->
Full experiment results are pending.
<!-- project4-results:end -->

The current-protocol figures and [generated summary](experiments/project4/results.md) are produced from the new CSVs by `python -m src.run_stats`. The five-slide [Marp deck](slides/project4.md) is refreshed by the same command. The project’s [checklist](CHECKLIST.md) records which experiments have actually run. The tracked tables in `experiments/results/` and images in `figures/` preserve the original exploratory results from before the protocol correction.

## Limitations

Local VAE neighbours reflect the learned decoder and can lose texture or class information. Pixel RMSE controls average image change and leaves differences in spatial structure unaccounted for. Conditional matching can exclude some images and variants. Deletion and insertion depend on the replacement baseline, and a strong classifier may assign high confidence to heavily edited inputs. The sanity-check reproduction uses a different architecture, domain and output space from Adebayo et al. FID is outside the evaluation because the matched comparison concerns local changes and the additional feature-extractor pass would add compute.
