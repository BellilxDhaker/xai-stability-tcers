from pathlib import Path
import json
import os

import pandas as pd

from src.evaluation.common import LEARNED_METHODS

LABELS = {"grad_cam": "Grad-CAM", "integrated_gradients": "IG", "occlusion": "Occlusion",
          "random": "Random", "constant": "Constant"}
ROOT = Path(__file__).resolve().parent.parent


def number(value):
    return f"{value:.3f}" if pd.notna(value) else "—"


def estimate(row):
    if pd.isna(row["mean"]):
        return f"— (n={row['n']})"
    return (f"{number(row['mean'])} [{number(row['ci_low'])}, {number(row['ci_high'])}] "
            f"(n={row['n']})")


def lookup(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    return estimate(frame.iloc[0]) if len(frame) else "—"


def mean_value(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    return number(frame.iloc[0]["mean"]) if len(frame) else "—"


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *[line(row) for row in rows]])


def order(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    return [LABELS[method] for method in frame[frame.method.isin(LEARNED_METHODS)]
            .dropna(subset=["mean"]).sort_values("mean", ascending=False).method]


def ranking(names):
    return " → ".join(names) or "No eligible images"


def render_reports(config, tables):
    scores, faith = tables["tcers_summary"], tables["faithfulness_summary"]
    sanity, sensitivity = tables["sanity_summary"], tables["sanity_parameter_sensitivity"]
    classic_order = order(scores, arm="decoded_classic_matched")
    vae_order = order(scores, arm="vae_matched")
    sanity_order = order(sensitivity, mode="cascading")
    matching = tables["matched_pairs"]
    population = config.test_samples
    fit_mode = "Smoke run · Workflow verification only" if config.quick else "Full protocol"
    class_metrics = config.output_root / "test_metrics.json"
    classes = json.loads(class_metrics.read_text(encoding="utf-8")) if class_metrics.exists() else {}
    class_sentence = (f"Classifier test accuracy {classes['accuracy']:.1%} on {classes['n']} images."
                      if classes else "Classifier test metrics are pending.")

    tcers_table = table(("Method", "Classic on raw", "Classic on decoded, matched", "VAE, matched"), [
        (label, *[lookup(scores, arm=arm, method=method) for arm in
                  ("classic", "decoded_classic_matched", "vae_matched")])
        for method, label in LABELS.items()
    ])
    auc_table = table(("Method", "Decoded deletion ↓", "Decoded insertion ↑"), [
        (label, *[lookup(faith, method=method, domain="decoded", baseline=config.faithfulness_baselines[0], metric=metric)
                  for metric in ("deletion_auc", "insertion_auc")])
        for method, label in LABELS.items()
    ])
    contribution = table(("Method", "Classic TCERS", "VAE TCERS", "Deletion AUC ↓", "Insertion AUC ↑"), [
        (label, *[mean_value(scores, arm=arm, method=method) for arm in ("decoded_classic_matched", "vae_matched")],
         *[mean_value(faith, method=method, domain="decoded", baseline=config.faithfulness_baselines[0], metric=metric)
           for metric in ("deletion_auc", "insertion_auc")]) for method, label in LABELS.items()
    ])
    sanity_table = table(("Method", "FC signed ρ", "Stem signed ρ", "Stem magnitude ρ", "Stem SSIM", "Stem HOG"), [
        (label, lookup(sanity, method=method, mode="cascading", layer="fc", metric="spearman_signed"),
         *[lookup(sanity, method=method, mode="cascading", layer="stem", metric=metric)
           for metric in ("spearman_signed", "spearman_abs", "ssim", "hog")])
        for method, label in LABELS.items()
    ])
    if classic_order and vae_order:
        rq1 = ("The point-estimate ranking changes between the two matched arms."
               if classic_order != vae_order else "The point-estimate ranking is the same in both matched arms.")
    else:
        rq1 = "The matched arms have insufficient overlap for a ranking."
    tests = tables["matched_arm_tests"]
    match_tests = table(("Method", "n", "Mean Δ TCERS", "95% CI", "Rank-biserial", "Holm p"), [
        (LABELS[row.method], row.n, number(row.mean_difference), f"[{number(row.ci_low)}, {number(row.ci_high)}]",
         number(row.rank_biserial), number(row.p_holm)) for row in tests.itertuples()
    ])
    corr_table = tables["stability_faithfulness_correlations"]
    corr_table = corr_table[(corr_table.arm == "vae_matched") &
                            (corr_table.baseline == config.faithfulness_baselines[0])]
    correlations = table(("Method", "Outcome", "n", "ρ [95% CI]", "Holm p"), [
        (LABELS[row.method], row.metric, row.n, f"{number(row.rho)} [{number(row.ci_low)}, {number(row.ci_high)}]",
         number(row.p_holm)) for row in corr_table.itertuples()
    ])
    if config.quick:
        statement = "The reduced training and evaluation settings make these outputs suitable for workflow checks."
    else:
        statement = rq1
    sensitivity_agreement = "The two point-estimate rankings cannot yet be compared."
    if sanity_order and vae_order:
        sensitivity_agreement = ("The two point-estimate rankings agree." if sanity_order == vae_order
                                 else "The two point-estimate rankings differ.")
    fit_runs = config.output_root / "vae_history.csv"
    if fit_runs.exists():
        history = pd.read_csv(fit_runs)
        best = history.loc[history.groupby("seed").validation_loss.idxmin()]
        loss = table(("Seed", "Epoch", "Validation MSE", "Validation KL"), [
            (row.seed, row.epoch, number(row.validation_mse), number(row.validation_kl))
            for row in best.itertuples()
        ])
    else:
        loss = "VAE training history is pending."
    coverage = f"The matcher retained {len(matching)} pairs across {matching['sample'].nunique()} of {population} images."
    rmse = tables["match_summary"]
    distances = (f"Classic RMSE {lookup(rmse, metric='pixel_rmse_classic')} · "
                 f"VAE RMSE {lookup(rmse, metric='pixel_rmse_vae')}.")
    agreement = lookup(tables["reconstruction_summary"], metric="class_preserved")
    report = f"""# Project 4 results

{fit_mode}. {class_sentence}

## RQ1 · Classic and VAE stability

{coverage} {distances} Classifier agreement between original and reconstructed images was {agreement}.

{tcers_table}

Matched classic order · {ranking(classic_order)}.

Matched VAE order · {ranking(vae_order)}.

{statement} Paired differences below subtract matched classic TCERS from matched VAE TCERS. Image-level intervals and Holm-adjusted tests quantify uncertainty.

{match_tests}

### VAE validation at the selected epoch

{loss}

## RQ2 · Stability and faithfulness

The following table uses the `{config.faithfulness_baselines[0]}` baseline and decoded images. The CSVs include raw images and every configured baseline. A constant explanation induces a seeded random order under faithfulness evaluation. Its TCERS and AUC estimate different properties.

{auc_table}

Spearman correlations for matched VAE TCERS and faithfulness are shown below. AUC directions differ, with lower deletion and higher insertion indicating better faithfulness. Correlations use image means from matching seeds.

{correlations}

## RQ3 · Adebayo parameter randomization

The cascading estimates below focus on the first and last ResNet18 resets. See the [layerwise curves](figures/sanity_checks.png) for both variants and [the published comparison protocol](../../docs/sanity_protocol.md) for architecture and metric differences.

{sanity_table}

Adebayo et al. found that plain Grad-CAM changed with parameter resets. Signed IG rank similarity dropped after upper-layer resets while magnitude retained more structure. Sliding-window Occlusion extends their quantitative comparison.

Cascading parameter sensitivity (mean of 1 − signed Spearman across reset stages) · {ranking(sanity_order)}.

Matched VAE stability · {ranking(vae_order)}.

{sensitivity_agreement} The ordering is descriptive for the three explanation methods. Adebayo et al. also reported an approximately zero signed and magnitude Spearman correlation between IG maps and Gaussian random masks. Here random and constant controls appear alongside the methods for every layer and metric. [Adebayo et al., Figures 2–4 and Appendix B](https://arxiv.org/html/1810.03292)
"""
    return report, {
        "@@MODE@@": fit_mode, "@@CLASSIFIER@@": class_sentence,
        "@@SANITY_VISUAL@@": "![width:1000px](@@FIGURES@@/sanity_signed.png)",
        "@@CONTRIBUTION@@": contribution, "@@MATCH@@": coverage,
        "@@RANK@@": statement, "@@SENSITIVITY@@": ranking(sanity_order),
        "@@VAE_VISUAL@@": (f"![width:380px](@@FIGURES@@/vae_grid_{config.evaluation_seeds[0]}.png)\n\n"
                            "> Validation images, reconstructions and prior samples appear from top to bottom."
                            if fit_runs.exists() else "> The VAE reconstruction grid is pending."),
    }


def write_reports(config, tables):
    report, values = render_reports(config, tables)
    (config.output_root / "results.md").write_text(report, encoding="utf-8")
    template = (ROOT / "slides" / "project4.template.md").read_text(encoding="utf-8")
    destination = config.output_root / "project4.md" if config.quick else ROOT / "slides" / "project4.md"
    figure_path = Path(os.path.relpath(config.figures.resolve(), destination.parent.resolve())).as_posix()
    values["@@FIGURES@@"] = figure_path
    for token, value in values.items():
        template = template.replace(token, value)
    destination.write_text(template, encoding="utf-8")
    if not config.quick:
        readme = ROOT / "README.md"
        start, end = "<!-- project4-results:start -->", "<!-- project4-results:end -->"
        prefix, found, tail = readme.read_text(encoding="utf-8").partition(start)
        _, matched_end, suffix = tail.partition(end)
        if found and matched_end:
            summary = f"{values['@@CLASSIFIER@@']} {values['@@MATCH@@']} {values['@@RANK@@']}"
            readme.write_text(f"{prefix}{start}\n{summary}\n{end}{suffix}", encoding="utf-8")
