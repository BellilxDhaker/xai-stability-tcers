from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from skimage.feature import hog
from skimage.metrics import structural_similarity
from torch import nn

from src.evaluation.common import METHODS, ExplanationMaps, make_generator
from src.metrics.stability import normalize_map, rank_similarity

LAYERS = ("fc", "layer4", "layer3", "layer2", "layer1", "stem")
SANITY_METRICS = ("spearman_signed", "spearman_abs", "ssim", "hog")


def reset_stage(model, name, seed):
    modules = (model.backbone.conv1, model.backbone.bn1) if name == "stem" else (getattr(model.backbone, name),)
    generator = make_generator("cpu", seed, name)
    with torch.no_grad():
        for root in modules:
            for layer in root.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    values = torch.empty_like(layer.weight, device="cpu")
                    nn.init.trunc_normal_(values, std=0.02, a=-0.04, b=0.04, generator=generator)
                    layer.weight.copy_(values)
                    if layer.bias is not None:
                        layer.bias.zero_()
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.reset_parameters()


def pearson(a, b):
    a_centered, b_centered = a - a.mean(), b - b.mean()
    product = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if product == 0:
        return float(np.array_equal(a, b))
    return float(np.clip(np.dot(a_centered, b_centered) / product, -1, 1))


def sanity_representation(values):
    scaled = normalize_map(values)
    descriptor = hog(scaled, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return scaled, values.flatten(), np.abs(values).flatten(), descriptor


def sanity_similarity(a, b):
    return {
        "spearman_signed": rank_similarity(a[1], b[1]),
        "spearman_abs": rank_similarity(a[2], b[2]),
        "ssim": float(structural_similarity(a[0], b[0], data_range=1.0, win_size=5)),
        "hog": pearson(a[3], b[3]),
    }


def randomized_models(model, mode, seed):
    randomized = deepcopy(model) if mode == "cascading" else None
    for depth, name in enumerate(LAYERS, start=1):
        if mode == "independent":
            randomized = deepcopy(model)
        reset_stage(randomized, name, seed)
        yield depth, name, randomized.eval()


def sanity_reference(config, provider, loader, model, target_device):
    references = {}
    for images, _, indices in loader:
        ids = tuple(indices.tolist())
        images = images.to(target_device)
        with torch.no_grad():
            targets = model(images).argmax(1)
        maps = provider(images, targets, signed=True, random_seed=("trained", ids))
        representations = {name: [sanity_representation(value[0].numpy()) for value in values]
                           for name, values in maps.items()}
        references[ids] = (targets.cpu(), representations)
    return references


def run_sanity(config, model, checkpoint_key, loader, target_device):
    trained_provider = ExplanationMaps(model, config, checkpoint_key)
    references = sanity_reference(config, trained_provider, loader, model, target_device)
    rows = []
    for seed in config.evaluation_seeds:
        for mode in ("cascading", "independent"):
            for ids, (_, representations) in references.items():
                rows.extend({"sample": sample, "seed": seed, "mode": mode, "depth": 0, "layer": "trained",
                             "method": method, **dict.fromkeys(SANITY_METRICS, 1.0)}
                            for sample in ids for method in representations)
            for depth, name, randomized in randomized_models(model, mode, seed):
                provider_mode = "shared" if depth == 1 else mode
                provider = ExplanationMaps(randomized, config, (checkpoint_key, seed, provider_mode, depth, "reset-v1"))
                for images, _, indices in loader:
                    ids = tuple(indices.tolist())
                    targets, original = references[ids]
                    maps = provider(images.to(target_device), targets.to(target_device), signed=True,
                                    random_seed=(seed, ids, provider_mode, depth))
                    for method in METHODS:
                        for i, sample in enumerate(ids):
                            candidate = sanity_representation(maps[method][i, 0].numpy())
                            rows.append({"sample": sample, "seed": seed, "mode": mode, "depth": depth, "layer": name,
                                         "method": method, **sanity_similarity(original[method][i], candidate)})
                print(f"Sanity {mode}, seed {seed}, reset {name}", flush=True)
    return pd.DataFrame(rows)
