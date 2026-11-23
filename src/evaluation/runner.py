from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from src.evaluation.common import METHODS, make_generator
from src.metrics.stability import TCERS_METRICS, similarity_metrics, tcers
from src.perturbations.transforms import make_perturbations
from src.perturbations.vae_perturbations import latent_neighbourhood

SIMILARITY_METRICS = (*TCERS_METRICS, "iou")
WEIGHTS = {"equal": (1/3, 1/3, 1/3), "ssim_heavy": (0.6, 0.2, 0.2),
           "spearman_heavy": (0.2, 0.6, 0.2), "cosine_heavy": (0.2, 0.2, 0.6)}


@torch.no_grad()
def predict(model, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    confidence, labels = model(images).softmax(1).max(1)
    return labels.cpu().numpy(), confidence.cpu().numpy()


@dataclass
class Variant:
    name: str
    parameter: float
    pixels: torch.Tensor
    targets: torch.Tensor
    confidence: np.ndarray
    distance: np.ndarray
    geometry: object = None


def pixel_rmse(reference, candidate):
    return (reference - candidate).square().mean(dim=(1, 2, 3)).sqrt().cpu().numpy()


@torch.no_grad()
def classify_variants(model, reference, candidates, forward_batch_size):
    groups = max(1, forward_batch_size // len(reference))
    variants = []
    for start in range(0, len(candidates), groups):
        chunk = candidates[start:start + groups]
        logits = model(torch.cat([pixels for _, _, pixels, _ in chunk]))
        confidence, targets = logits.softmax(1).max(1)
        for (name, parameter, pixels, geometry), labels, conf in zip(
                chunk, targets.split(len(reference)), confidence.split(len(reference))):
            variants.append(Variant(name, parameter, pixels, labels, conf.cpu().numpy(),
                                    pixel_rmse(reference, pixels), geometry))
    return variants


def classic_candidates(reference, perturbations, seed, ids):
    candidates = []
    for variant in perturbations:
        generator = make_generator(reference.device, seed, tuple(ids), variant.name)
        candidates.append((variant.name, variant.parameter, variant.apply(reference, generator), variant))
    return candidates


def metric_maps(reference, candidate, geometry, metric_size):
    support = None
    if geometry is not None:
        reference, support = geometry.align(reference)
    if metric_size is not None:
        shape = (metric_size, metric_size)
        reference = F.adaptive_avg_pool2d(reference, shape)
        candidate = F.adaptive_avg_pool2d(candidate, shape)
        if support is not None:
            support = F.adaptive_avg_pool2d(support.float()[None, None], shape)[0, 0] == 1
    return reference[:, 0].numpy(), candidate[:, 0].numpy(), None if support is None else support.numpy()


def match_variants(reference_targets, classic, latent, caliper):
    """Match without replacement within each image, maximizing the eligible pair count."""
    classic_matches = np.full((len(classic), len(reference_targets)), -1)
    latent_matches = np.full((len(latent), len(reference_targets)), -1)
    classic_targets = np.stack([v.targets.cpu().numpy() for v in classic])
    latent_targets = np.stack([v.targets.cpu().numpy() for v in latent])
    for image, label in enumerate(reference_targets.cpu().tolist()):
        classic_distance = np.array([variant.distance[image] for variant in classic])
        latent_distance = np.array([variant.distance[image] for variant in latent])
        distances = np.abs(latent_distance[:, None] - classic_distance[None, :])
        eligible = ((distances <= caliper) & (classic_targets[:, image] == label)[None, :]
                    & (latent_targets[:, image] == label)[:, None])
        penalty = (len(latent) + 1) * (1 + caliper)
        costs = np.where(eligible, distances, penalty)
        latent_index, classic_index = linear_sum_assignment(costs)
        for li, ci in zip(latent_index, classic_index):
            if eligible[li, ci]:
                classic_matches[ci, image] = li
                latent_matches[li, image] = li
    return classic_matches, latent_matches


def evaluate_arm(reference_maps, targets, confidence, variants, matches, ids, seed, arm,
                 map_provider, metric_size):
    labels = targets.cpu().numpy()
    for index, variant in enumerate(variants):
        candidate_labels = variant.targets.cpu().numpy()
        unchanged = candidate_labels == labels
        eligible = unchanged if arm != "decoded_classic" else unchanged & (matches[index] >= 0)
        positions = np.flatnonzero(eligible)
        changed_maps = {}
        if len(positions):
            candidates = map_provider(variant.pixels[positions], targets[positions],
                                      random_seed=(seed, tuple(ids), arm, variant.name, variant.parameter))
            changed_maps = {method: metric_maps(values[positions], candidates[method], variant.geometry, metric_size)
                            for method, values in reference_maps.items()}
        metric_positions = {position: offset for offset, position in enumerate(positions)}
        for position, sample in enumerate(ids):
            for method in METHODS:
                row = {
                    "sample": sample, "seed": seed, "arm": arm, "method": method,
                    "transformation": variant.name, "parameter": variant.parameter,
                    "prediction_unchanged": bool(unchanged[position]),
                    "reference_prediction": labels[position], "transformed_prediction": candidate_labels[position],
                    "original_confidence": confidence[position], "transformed_confidence": variant.confidence[position],
                    "pixel_rmse": variant.distance[position], "match": matches[index, position],
                    **dict.fromkeys(SIMILARITY_METRICS, np.nan),
                }
                if position in metric_positions:
                    offset = metric_positions[position]
                    a, b, mask = changed_maps[method]
                    row.update(similarity_metrics(a[offset], b[offset], mask=mask))
                yield row


def run_stability(config, model, vaes, loader, map_provider, target_device):
    rows, reconstructions = [], []
    perturbations = make_perturbations(config.perturbations)
    for batch, (images, labels, indices) in enumerate(loader, start=1):
        ids = tuple(indices.tolist())
        images = images.to(target_device)
        raw_targets, raw_confidence = predict(model, images)
        raw_targets = torch.as_tensor(raw_targets, device=target_device)
        for seed, vae in vaes.items():
            raw_maps = map_provider(images, raw_targets, random_seed=(seed, ids, "raw"))
            classic = classify_variants(model, images, classic_candidates(images, perturbations, seed, ids),
                                        config.forward_batch_size)
            rows.extend(evaluate_arm(raw_maps, raw_targets, raw_confidence, classic,
                                     np.full((len(classic), len(images)), -1), ids, seed, "classic",
                                     map_provider, config.metric_size))
            decoded, neighbours = latent_neighbourhood(
                vae, images, config.vae_size, config.latent_sigmas, make_generator(target_device, seed, ids, "latent"),
            )
            decoded_targets, decoded_confidence = predict(model, decoded)
            reconstruction_distance = pixel_rmse(images, decoded)
            reconstructions.extend({"sample": sample, "seed": seed, "label": int(labels[i]),
                                    "raw_prediction": int(raw_targets[i]), "decoded_prediction": decoded_targets[i],
                                    "reconstruction_pixel_rmse": reconstruction_distance[i]}
                                   for i, sample in enumerate(ids))
            decoded_targets = torch.as_tensor(decoded_targets, device=target_device)
            decoded_maps = map_provider(decoded, decoded_targets, random_seed=(seed, ids, "decoded"))
            decoded_classic = classify_variants(
                model, decoded, classic_candidates(decoded, perturbations, seed, ids), config.forward_batch_size,
            )
            latent = classify_variants(model, decoded, [
                ("latent", sigma, pixels, None) for sigma, pixels in zip(config.latent_sigmas, neighbours)
            ], config.forward_batch_size)
            classic_matches, latent_matches = match_variants(decoded_targets, decoded_classic, latent, config.match_caliper)
            for arm, variants, matches in (("decoded_classic", decoded_classic, classic_matches),
                                            ("vae", latent, latent_matches)):
                rows.extend(evaluate_arm(decoded_maps, decoded_targets, decoded_confidence, variants, matches, ids,
                                         seed, arm, map_provider, config.metric_size))
        print(f"Stability batch {batch}/{len(loader)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(reconstructions)


def aggregate_tcers(results: pd.DataFrame, min_valid: int = 1, weights=WEIGHTS["equal"]) -> pd.DataFrame:
    keys = [key for key in ("sample", "method", "seed", "arm") if key in results]
    rows = []
    for group, frame in results.groupby(keys, sort=False):
        valid = frame[frame.prediction_unchanged]
        summary = tcers(valid[list(TCERS_METRICS)].to_dict("records"), weights, min_valid, len(frame))
        rows.append({**dict(zip(keys, group)), **summary,
                     "prediction_consistency": frame.prediction_unchanged.mean()})
    return pd.DataFrame(rows)
