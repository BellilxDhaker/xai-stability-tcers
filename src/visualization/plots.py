from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.report import LABELS


def save_figure(figure, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_training_history(history: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    history.plot(x="epoch", y=["train_loss", "validation_loss"], ax=axes[0], title="Loss")
    history.plot(x="epoch", y=["train_accuracy", "validation_accuracy"], ax=axes[1], title="Accuracy")
    save_figure(figure, output)


def plot_stability(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=results, x="transformation", y="ssim", hue="method", ax=axis)
    axis.set(title="Explanation stability by transformation", ylabel="SSIM")
    save_figure(figure, output)


def plot_lines(axis, data, x):
    for method, label in LABELS.items():
        group = data[data.method == method].sort_values(x)
        if len(group):
            line, = axis.plot(group[x], group["mean"], marker="o", markersize=3, label=label)
            axis.fill_between(group[x], group.ci_low, group.ci_high, color=line.get_color(), alpha=.12)
    axis.grid(alpha=.2)


def plot_sanity(data, metrics, output):
    modes = ("cascading", "independent")
    figure, axes = plt.subplots(len(metrics), len(modes), figsize=(10, 2.55 * len(metrics) + .4), squeeze=False)
    ticks = data[["depth", "layer"]].drop_duplicates().sort_values("depth")
    for i, metric in enumerate(metrics):
        for j, mode in enumerate(modes):
            axis = axes[i, j]
            plot_lines(axis, data[(data["mode"] == mode) & (data.metric == metric)], "depth")
            axis.set(ylim=(-1.02, 1.02), ylabel=metric.replace("_", " "),
                     xticks=ticks.depth, xticklabels=ticks.layer if i == len(metrics) - 1 else [])
            if i == 0:
                axis.set_title(mode.title())
    axes[0, 0].legend(ncol=3, loc="lower left", fontsize=7)
    save_figure(figure, output)


def plot_faithfulness(data, config):
    baseline = config.faithfulness_baselines[0]
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for i, domain in enumerate(("raw", "decoded")):
        for j, metric in enumerate(("deletion", "insertion")):
            axis = axes[i, j]
            chosen = (data.domain == domain) & (data.metric == metric) & (data.baseline == baseline)
            plot_lines(axis, data[chosen], "fraction")
            axis.set(title=f"{domain.title()} · {metric} ({baseline})", xlabel="Fraction of pixels changed",
                     ylabel="Target confidence")
    axes[0, 0].legend(fontsize=7)
    save_figure(figure, config.figures / "faithfulness.png")


def plot_tcers(data, config):
    arms = ("classic", "decoded_classic_matched", "vae_matched")
    figure, axes = plt.subplots(1, len(arms), figsize=(11, 3.7), sharex=True)
    for axis, arm in zip(axes, arms):
        rows = data[(data.arm == arm) & data["mean"].notna()].set_index("method").reindex(LABELS).dropna()
        if len(rows):
            axis.errorbar(rows["mean"], [LABELS[method] for method in rows.index],
                          xerr=[(rows["mean"] - rows.ci_low).clip(lower=0), (rows.ci_high - rows["mean"]).clip(lower=0)],
                          fmt="o", capsize=3)
        axis.set(xlim=(0, 1.04), xlabel="TCERS", title=arm.replace("_", " ").title())
        axis.grid(axis="x", alpha=.2)
    save_figure(figure, config.figures / "tcers.png")


def plot_experiment_results(config, tables):
    sanity = tables["sanity_summary"]
    plot_sanity(sanity, ("spearman_signed", "spearman_abs", "ssim", "hog"), config.figures / "sanity_checks.png")
    plot_sanity(sanity, ("spearman_signed",), config.figures / "sanity_signed.png")
    plot_faithfulness(tables["faithfulness_curve_summary"], config)
    plot_tcers(tables["tcers_summary"], config)
