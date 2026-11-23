import json

import numpy as np
import pandas as pd

from src.config import config_from_args
from src.evaluation.common import LEARNED_METHODS
from src.evaluation.runner import SIMILARITY_METRICS, WEIGHTS, aggregate_tcers
from src.evaluation.sanity_checks import SANITY_METRICS
from src.evaluation.statistics import faithfulness_correlations, paired_tests, summarize
from src.report import write_reports
from src.visualization.plots import plot_experiment_results


def read_results(config):
    root = config.output_root
    signatures = [json.loads((root / f"signature_{stage}.json").read_text(encoding="utf-8"))
                  for stage in ("stability", "faithfulness", "sanity")]
    if signatures[0].pop("vae_sha256") != signatures[1].pop("vae_sha256"):
        raise ValueError("Stability and faithfulness used different VAE checkpoints.")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("The evaluation stages used different classifiers, cohorts or attribution settings.")
    return [pd.read_csv(root / f"{name}.csv") for name in
            ("stability_raw", "reconstructions", "faithfulness_auc", "faithfulness_curves", "sanity_raw")]


def match_table(raw):
    fields = ["sample", "seed", "match", "transformation", "parameter", "pixel_rmse"]
    controls = raw.loc[(raw.arm == "decoded_classic") & (raw.match >= 0) & (raw.method == "constant"), fields]
    latent = raw.loc[(raw.arm == "vae") & (raw.match >= 0) & (raw.method == "constant"), fields]
    pairs = controls.merge(latent, on=["sample", "seed", "match"], suffixes=("_classic", "_vae"), validate="one_to_one")
    pairs["absolute_rmse_gap"] = np.abs(pairs.pixel_rmse_classic - pairs.pixel_rmse_vae)
    return pairs


def run(config):
    config.create_directories()
    config.save("stats")
    raw, reconstructions, faithfulness, curves, sanity = read_results(config)

    matched = raw[(raw.match >= 0) & raw.arm.isin(["decoded_classic", "vae"])].copy()
    matched["arm"] = matched.arm + "_matched"
    raw_with_matches = pd.concat([raw, matched], ignore_index=True)
    scores = aggregate_tcers(raw_with_matches)
    sensitivity = pd.concat([aggregate_tcers(raw_with_matches, weights=weights).assign(weighting=name)
                             for name, weights in WEIGHTS.items()], ignore_index=True)
    tcers_summary = summarize(scores, ["arm", "method"], ["tcers"], config)
    paired_scores = scores[scores.arm.isin(["decoded_classic_matched", "vae_matched"]) & scores.method.isin(LEARNED_METHODS)]
    interaction = paired_scores.pivot_table(index=["sample", "seed", "method"], columns="arm", values="tcers")
    interaction = interaction.reindex(columns=["vae_matched", "decoded_classic_matched"]).dropna()
    interaction["delta"] = interaction.vae_matched - interaction.decoded_classic_matched
    interaction = interaction.reset_index().assign(contrast="vae_minus_classic")
    prediction = raw[raw.method == "constant"].copy()
    prediction["prediction_unchanged"] = prediction.prediction_unchanged.astype(float)
    classic_consistency = prediction[prediction.arm == "classic"]
    pairs = match_table(raw)
    reconstructions["class_preserved"] = (reconstructions.raw_prediction == reconstructions.decoded_prediction).astype(float)
    reconstructions["decoded_correct"] = (reconstructions.decoded_prediction == reconstructions.label).astype(float)
    reconstructions["raw_correct"] = (reconstructions.raw_prediction == reconstructions.label).astype(float)
    reconstructions["domain"] = "decoded"

    sanity_summary = summarize(sanity, ["mode", "depth", "layer", "method"], SANITY_METRICS, config)
    sanity_sensitivity = sanity[sanity.depth > 0].copy()
    sanity_sensitivity["parameter_sensitivity"] = 1 - sanity_sensitivity.spearman_signed
    sanity_method = summarize(sanity_sensitivity, ["mode", "method"], ["parameter_sensitivity"], config)
    faith_summary = summarize(faithfulness, ["domain", "baseline", "method"], ["deletion_auc", "insertion_auc"], config)
    correlations = faithfulness_correlations(scores, faithfulness, config)
    tables = {
        "stability_summary": summarize(raw_with_matches, ["arm", "method", "transformation"], SIMILARITY_METRICS, config),
        "stability_by_parameter": summarize(raw_with_matches, ["arm", "method", "transformation", "parameter"],
                                            SIMILARITY_METRICS, config),
        "tcers_per_sample": scores,
        "tcers_summary": tcers_summary,
        "tcers_by_seed": summarize(scores, ["arm", "method", "seed"], ["tcers"], config),
        "tcers_weight_sensitivity": summarize(sensitivity, ["arm", "method", "weighting"], ["tcers"], config),
        "tcers_pairwise": paired_tests(scores, ["arm"], "method", "tcers", config),
        "matched_arm_tests": paired_tests(paired_scores, ["method"], "arm", "tcers", config,
                                         only=[("vae_matched", "decoded_classic_matched")]),
        "ranking_interactions": paired_tests(interaction, ["contrast"], "method", "delta", config),
        "matched_pairs": pairs,
        "match_summary": summarize(pairs.assign(domain="matched"), ["domain"],
                                   ["pixel_rmse_classic", "pixel_rmse_vae", "absolute_rmse_gap"], config),
        "prediction_consistency": summarize(prediction, ["arm", "transformation", "parameter"],
                                            ["prediction_unchanged", "pixel_rmse"], config),
        "prediction_consistency_pairwise": paired_tests(classic_consistency.assign(arm="classic"), ["arm"],
                                                        "transformation", "prediction_unchanged", config),
        "reconstruction_summary": summarize(reconstructions, ["domain"],
                                             ["reconstruction_pixel_rmse", "class_preserved",
                                              "decoded_correct", "raw_correct"], config),
        "faithfulness_summary": faith_summary,
        "faithfulness_curve_summary": summarize(curves, ["domain", "baseline", "method", "fraction"],
                                                ["deletion", "insertion"], config),
        "stability_faithfulness_correlations": correlations,
        "sanity_summary": sanity_summary,
        "sanity_by_seed": summarize(sanity, ["mode", "depth", "layer", "method", "seed"], SANITY_METRICS, config),
        "sanity_parameter_sensitivity": sanity_method,
        "sanity_pairwise": paired_tests(sanity_sensitivity, ["mode"], "method", "parameter_sensitivity", config),
    }
    for metric in ("deletion_auc", "insertion_auc"):
        tables[f"faithfulness_{metric}_pairwise"] = paired_tests(faithfulness, ["domain", "baseline"], "method", metric, config)
    for name, table in tables.items():
        table.to_csv(config.output_root / f"{name}.csv", index=False)
    plot_experiment_results(config, tables)
    write_reports(config, tables)


def main():
    run(config_from_args("Summarize evaluation CSVs with image bootstrap and paired Holm tests."))


if __name__ == "__main__":
    main()
