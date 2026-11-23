# Project 4 checklist

| Requirement | Evidence |
| --- | --- |
| Part 1 · Scientific problem, motivation, objectives and RQ1–RQ3 | [Problem statement](docs/problem_statement.md), [README](README.md), slide 1 |
| Part 2 · Recent sources, comparison and identified gap | [State of the art](docs/state_of_the_art.md), [references](docs/references.md), slide 2 |
| Part 3 · Published implementation and protocol | [Sanity checks](src/evaluation/sanity_checks.py), [CLI](src/run_sanity_checks.py), [published comparison](docs/sanity_protocol.md), slide 3 |
| Part 3 · Comparison with measured results | Generated `sanity_summary.csv`, `sanity_checks.png` and `results.md` |
| Part 4 · VAE train split, reconstruction, KL and generated samples | [VAE](src/models/vae.py), [training](src/train_vae.py), generated `vae_history.csv` and `vae_grid_*.png` |
| Part 4 · Matched stability and faithfulness | [Stability](src/evaluation/runner.py), [latent neighbours](src/perturbations/vae_perturbations.py), [faithfulness](src/evaluation/faithfulness.py), slides 4–5 |
| Part 4 · Paired statistics, weight sensitivity and RQ outputs | [Statistics CLI](src/run_stats.py), [analysis](src/evaluation/statistics.py), generated CSVs and result summary |
| Code, configuration and tests | [Source](src/), [configuration](src/config.py), [pinned packages](requirements.txt), [tests](tests/) |
| Reproducible notebooks and five slides maximum | [Notebooks](notebooks/), [Marp deck](slides/project4.md) |
| Selected papers and dataset | [Publisher copies](papers/README.md), [Oxford archives, licence and split statistics](data/README.md) |

## Verification status

- `pytest -q` passed 13 tests. All six CLI stages completed in CPU smoke mode with Oxford-IIIT Pet images, including a separate three-seed smoke run with multiple batches. Local smoke artifacts are under `experiments/project4-quick/`.
- The five-slide draft and the populated smoke deck were rendered as Marp slides. The full report and deck mark research results as pending.
- The four selected PDFs are available locally. The dataset includes every official annotated trainval and test image. Both downloads are excluded from Git. The [generated split table](experiments/project4/dataset_splits.csv) verifies the 300-image stratified cohort.
- TODO Run the full cohort and three seeds, then generate the research report and deck from those CSVs. The CPU smoke runs verify the workflow at reduced settings. The full faithfulness stage alone would take hours at the measured CPU throughput.
- TODO Time the full pipeline on a Colab T4.
- FID and latent counterfactual optimization are optional and outside the implemented scope.
