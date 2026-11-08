from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.nanmin(values), np.nanmax(values)
    return (values - low) / max(high - low, 1e-12)


def similarity_metrics(reference: np.ndarray, candidate: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    a, b = normalize_map(reference), normalize_map(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Explanation maps must have equal shapes, got {a.shape} and {b.shape}.")
    flat_a, flat_b = a.ravel(), b.ravel()
    rho = spearmanr(flat_a, flat_b).statistic
    cosine = float(np.dot(flat_a, flat_b) /
                   (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12))
    intersection = np.logical_and(a >= threshold, b >= threshold).sum()
    union = np.logical_or(a >= threshold, b >= threshold).sum()
    return {
        "ssim": float(structural_similarity(a, b, data_range=1.0)),
        "iou": float(intersection / max(union, 1)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "cosine": cosine,
    }


def tcers(scores: Sequence[Dict[str, float]], weights=(1 / 3, 1 / 3, 1 / 3),
          min_valid: int = 1, total_transformations: int | None = None) -> Dict[str, float]:
    total = len(scores) if total_transformations is None else total_transformations
    if len(scores) < min_valid:
        return {"tcers": float("nan"), "valid_transformations": len(scores),
                "excluded_transformations": max(total - len(scores), 0)}
    if len(weights) != 3 or not np.isclose(sum(weights), 1.0):
        raise ValueError("TCERS weights must contain three values summing to one.")
    values = [weights[0] * item["ssim"] + weights[1] * item["spearman"] +
              weights[2] * item["cosine"] for item in scores]
    return {"tcers": float(np.mean(values)), "valid_transformations": len(scores),
            "excluded_transformations": max(total - len(scores), 0)}


def prediction_consistency(original: Iterable[int], transformed: Iterable[int]) -> float:
    original_array, transformed_array = np.asarray(list(original)), np.asarray(list(transformed))
    if original_array.shape != transformed_array.shape or original_array.size == 0:
        raise ValueError("Prediction arrays must be non-empty and have equal shape.")
    return float(np.mean(original_array == transformed_array))
