from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def rank_similarity(a, b) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float(np.array_equal(a, b))
    return float(np.clip(spearmanr(a, b).statistic, -1, 1))


def similarity_metrics(reference: np.ndarray, candidate: np.ndarray, threshold: float = 0.5,
                       mask: np.ndarray | None = None) -> dict[str, float]:
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("Explanation maps must be two-dimensional and have equal shapes.")
    support = np.ones(reference.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    window = min(7, min(reference.shape))
    window -= (window + 1) % 2
    valid_windows = binary_erosion(support, structure=np.ones((window, window), dtype=bool))
    if not valid_windows.any():
        raise ValueError("The overlap must contain a full SSIM window.")
    a, b = reference[support], candidate[support]
    # Float32 resampling can introduce rounding variation in a flat map.
    a = np.full_like(a, a[0]) if np.ptp(a) < 1e-6 else a
    b = np.full_like(b, b[0]) if np.ptp(b) < 1e-6 else b
    a, b = normalize_map(a), normalize_map(b)
    product = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = np.dot(a, b) / product if product > 0 else float(np.array_equal(a, b))
    foreground_a, foreground_b = a >= threshold, b >= threshold
    union = np.logical_or(foreground_a, foreground_b).sum()

    # Restrict SSIM to windows fully inside the common support.
    image_a, image_b = np.zeros(reference.shape), np.zeros(reference.shape)
    image_a[support], image_b[support] = a, b
    _, local_ssim = structural_similarity(image_a, image_b, data_range=1.0, win_size=window, full=True)
    return {
        "ssim": float(np.clip(local_ssim[valid_windows].mean(), -1, 1)),
        "iou": float(np.logical_and(foreground_a, foreground_b).sum() / union) if union else 1.0,
        "spearman": rank_similarity(a, b),
        "cosine": float(np.clip(cosine, 0, 1)),
    }


TCERS_METRICS = ("ssim", "spearman", "cosine")


def tcers(scores: Sequence[dict[str, float]], weights=(1 / 3, 1 / 3, 1 / 3),
          min_valid: int = 1, total_transformations: int | None = None) -> dict[str, float]:
    if len(weights) != 3 or np.any(np.asarray(weights) < 0) or not np.isclose(sum(weights), 1.0):
        raise ValueError("TCERS requires three nonnegative weights summing to one.")
    total = len(scores) if total_transformations is None else total_transformations
    if total < len(scores) or min_valid < 1:
        raise ValueError("Invalid TCERS transformation counts.")
    values = np.array([[row[key] for key in TCERS_METRICS] for row in scores]).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    valid = len(values)
    score = float("nan")
    if valid >= min_valid:
        values[:, :2] = (values[:, :2] + 1) / 2
        score = float(np.clip((values @ weights).mean(), 0, 1))
    return {"tcers": score, "valid_transformations": valid, "excluded_transformations": total - valid}


def prediction_consistency(original: Iterable[int], transformed: Iterable[int]) -> float:
    original_array, transformed_array = np.asarray(list(original)), np.asarray(list(transformed))
    if original_array.shape != transformed_array.shape or original_array.size == 0:
        raise ValueError("Prediction arrays must be non-empty and have equal shape.")
    return float(np.mean(original_array == transformed_array))
