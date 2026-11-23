from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, shapiro, spearmanr, wilcoxon


def mean_ci(values, repeats, rng):
    if len(values) == 0:
        return (np.nan, np.nan)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975])


def summarize(frame, groups, metrics, config):
    rng = np.random.default_rng(config.seed)
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        for metric in metrics:
            # The independent resampling unit is the image, across seeds and variants.
            values = group.groupby("sample")[metric].mean().dropna().to_numpy()
            low, high = mean_ci(values, config.bootstrap_repeats, rng)
            rows.append({**dict(zip(groups, keys)), "metric": metric, "n": len(values),
                         "mean": values.mean() if len(values) else np.nan,
                         "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows, columns=[*groups, "metric", "n", "mean", "std", "ci_low", "ci_high"])


def holm(frame, groups):
    frame["p_holm"] = np.nan
    for _, group in frame.groupby(groups, sort=False):
        ordered = group.p.dropna().sort_values()
        adjusted = np.maximum.accumulate(ordered.to_numpy() * np.arange(len(ordered), 0, -1)).clip(0, 1)
        frame.loc[ordered.index, "p_holm"] = adjusted
    return frame


def paired_difference(paired, left, right, config, rng):
    differences = (paired[left] - paired[right]).dropna().to_numpy()
    count = len(differences)
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero))
    low, high = mean_ci(differences, config.bootstrap_repeats, rng)
    return {
        "left": left, "right": right, "n": count,
        "mean_difference": differences.mean() if count else np.nan,
        "ci_low": low, "ci_high": high,
        "rank_biserial": np.dot(ranks, np.sign(nonzero)) / ranks.sum() if len(ranks) else (0.0 if count else np.nan),
        "p": wilcoxon(differences).pvalue if len(nonzero) else (1.0 if count else np.nan),
        "shapiro_p": shapiro(differences).pvalue if 3 <= count <= 5000 and np.ptp(differences) else np.nan,
    }


def paired_tests(frame, groups, contrast, metric, config, only=None):
    rng = np.random.default_rng(config.seed)
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        paired = group.pivot_table(index="sample", columns=contrast, values=metric, aggfunc="mean")
        choices = list(combinations(paired.columns, 2)) if only is None else only
        for left, right in choices:
            if left in paired and right in paired:
                rows.append({**dict(zip(groups, keys)), **paired_difference(paired, left, right, config, rng)})
    columns = [*groups, "left", "right", "n", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p", "shapiro_p"]
    return holm(pd.DataFrame(rows, columns=columns), groups)


def rank_ci(x, y, config, rng):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    observed = spearmanr(x, y)
    indices = rng.integers(0, len(x), (config.bootstrap_repeats, len(x)))
    x_rank, y_rank = rankdata(x[indices], axis=1), rankdata(y[indices], axis=1)
    x_rank -= x_rank.mean(axis=1, keepdims=True)
    y_rank -= y_rank.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x_rank, axis=1) * np.linalg.norm(y_rank, axis=1)
    valid = norm > 0
    estimates = (x_rank[valid] * y_rank[valid]).sum(axis=1) / norm[valid]
    low, high = np.quantile(estimates, [0.025, 0.975]) if len(estimates) else (np.nan, np.nan)
    return observed.statistic, observed.pvalue, low, high


def faithfulness_correlations(tcers, auc, config):
    rng = np.random.default_rng(config.seed)
    source = tcers.copy()
    source["domain"] = source.arm.map({"classic": "raw", "decoded_classic_matched": "decoded",
                                        "vae_matched": "decoded"})
    joined = source.merge(auc, on=["sample", "seed", "method", "domain"], how="inner", validate="many_to_many")
    groups = ["arm", "method", "baseline"]
    rows = []
    for (arm, method, baseline), group in joined.groupby(groups):
        for metric in ("deletion_auc", "insertion_auc"):
            values = group.dropna(subset=["tcers", metric]).groupby("sample")[["tcers", metric]].mean()
            rho, p, low, high = rank_ci(values.tcers.to_numpy(), values[metric].to_numpy(), config, rng)
            rows.append({"arm": arm, "method": method, "baseline": baseline, "metric": metric,
                         "n": len(values), "rho": rho, "p": p, "ci_low": low, "ci_high": high})
    return holm(pd.DataFrame(rows, columns=[*groups, "metric", "n", "rho", "p", "ci_low", "ci_high"]), ["arm"])
