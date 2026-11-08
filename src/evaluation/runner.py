from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from src.metrics.stability import similarity_metrics, tcers


@torch.no_grad()
def predict(model, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = model(images)
    probabilities = logits.softmax(1)
    return probabilities.argmax(1).cpu().numpy(), probabilities.max(1).values.cpu().numpy()


def evaluate_transformations(model, images: torch.Tensor, explain: Callable[[torch.Tensor], torch.Tensor],
                            perturbations: Iterable, device: torch.device, method: str = "unknown",
                            output_csv: Path | None = None) -> pd.DataFrame:
    model.eval()
    images = images.to(device)
    original_predictions, original_confidence = predict(model, images)
    original_maps = explain(images).detach().cpu().numpy()
    rows = []
    for perturbation in perturbations:
        transformed = perturbation.apply(images).to(device)
        if transformed.shape != images.shape:
            transformed = torch.nn.functional.interpolate(
                transformed, size=images.shape[-2:], mode="bilinear", align_corners=False)
        predictions, confidence = predict(model, transformed)
        transformed_maps = explain(transformed).detach().cpu().numpy()
        for index in range(images.size(0)):
            unchanged = bool(predictions[index] == original_predictions[index])
            row = {
                "sample": index, "transformation": perturbation.name,
                "parameter": perturbation.parameter, "method": method,
                "prediction_unchanged": unchanged,
                "original_confidence": original_confidence[index],
                "transformed_confidence": confidence[index],
            }
            if unchanged:
                row.update(similarity_metrics(original_maps[index, 0], transformed_maps[index, 0]))
            rows.append(row)
    results = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
    return results


def aggregate_tcers(results: pd.DataFrame, min_valid: int = 1) -> pd.DataFrame:
    valid = results[results["prediction_unchanged"]].copy()
    grouped = []
    for (sample, method), frame in valid.groupby(["sample", "method"], dropna=False):
        all_rows = results[(results["sample"] == sample) & (results["method"] == method)]
        summary = tcers(frame[["ssim", "spearman", "cosine"]].to_dict("records"),
                        min_valid=min_valid, total_transformations=len(all_rows))
        summary.update({"sample": sample, "method": method,
                        "prediction_consistency": all_rows["prediction_unchanged"].mean()})
        grouped.append(summary)
    return pd.DataFrame(grouped)
