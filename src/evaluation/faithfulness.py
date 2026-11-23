import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import gaussian_blur

from src.evaluation.common import make_generator


def baseline_image(images, baseline):
    if baseline == "mean":
        return images.mean(dim=(-2, -1), keepdim=True).expand_as(images)
    if baseline == "blur":
        window = min(31, 2 * min(images.shape[-2:]) - 1)
        return gaussian_blur(images, [window, window], [10.0, 10.0])
    raise ValueError(f"Unknown replacement baseline: {baseline}")


def pixel_ranks(maps, generator):
    flat = maps.flatten(1)
    # A seeded common permutation breaks ties without a raster-order advantage.
    permutation = torch.rand(flat.shape, device=flat.device, generator=generator).argsort(dim=1)
    order = flat.gather(1, permutation).argsort(dim=1, descending=True, stable=True)
    order = permutation.gather(1, order)
    return order.argsort(dim=1).reshape_as(maps)


@torch.no_grad()
def deletion_insertion(model, images, maps, target, baseline, steps, batch_size, generator):
    ranks = pixel_ranks(maps, generator)
    counts = torch.linspace(0, ranks[0].numel(), steps + 1, device=images.device).round().long().unique()
    fields = []
    for evaluation in ("deletion", "insertion"):
        values = []
        for chunk in counts.split(max(1, batch_size // len(images))):
            selected = ranks.unsqueeze(0) < chunk[:, None, None, None, None]
            if evaluation == "deletion":
                selected = ~selected
            perturbed = torch.where(selected, images, baseline).flatten(0, 1)
            scores = model(perturbed).softmax(1).gather(1, target.repeat(len(chunk))[:, None])
            values.append(scores.reshape(len(chunk), len(images)).cpu().numpy())
        fields.append(np.concatenate(values, axis=0))
    fractions = counts.cpu().numpy() / ranks[0].numel()
    return fractions, *fields


def evaluate_faithfulness(model, config, map_provider, images, ids, seed, domain):
    with torch.no_grad():
        target = model(images).argmax(1)
    maps = map_provider(images, target, random_seed=(seed, ids, domain))
    targets = target.cpu().tolist()
    auc_rows, curve_rows = [], []
    for baseline in config.faithfulness_baselines:
        replacement = baseline_image(images, baseline)
        for method, values in maps.items():
            fractions, deletion, insertion = deletion_insertion(
                model, images, values.to(images.device), target, replacement, config.faithfulness_steps,
                config.forward_batch_size, make_generator(images.device, seed, ids, domain, "ties"),
            )
            deletion_auc = np.trapezoid(deletion, fractions, axis=0)
            insertion_auc = np.trapezoid(insertion, fractions, axis=0)
            for i, sample in enumerate(ids):
                key = {"sample": sample, "seed": seed, "domain": domain,
                       "method": method, "baseline": baseline, "target": targets[i]}
                auc_rows.append({**key, "deletion_auc": deletion_auc[i], "insertion_auc": insertion_auc[i]})
                curve_rows.extend({**key, "fraction": fraction, "deletion": d, "insertion": ins}
                                  for fraction, d, ins in zip(fractions, deletion[:, i], insertion[:, i]))
    return pd.DataFrame(auc_rows), pd.DataFrame(curve_rows)
