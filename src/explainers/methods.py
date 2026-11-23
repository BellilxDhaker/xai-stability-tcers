import torch
from torch import nn
from torch.nn import functional as F


def _normalize(attribution: torch.Tensor) -> torch.Tensor:
    minimum = attribution.amin(dim=(-2, -1), keepdim=True)
    maximum = attribution.amax(dim=(-2, -1), keepdim=True)
    return (attribution - minimum) / (maximum - minimum).clamp_min(1e-12)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self.target_layer = model, target_layer

    @torch.enable_grad()
    def __call__(self, images: torch.Tensor, target: torch.Tensor | None = None, *, signed=False) -> torch.Tensor:
        activations = []
        handle = self.target_layer.register_forward_hook(lambda module, args, output: activations.append(output))
        try:
            logits = self.model(images.detach().requires_grad_(True))
        finally:
            handle.remove()
        labels = logits.argmax(1) if target is None else target
        activation = activations[0]
        gradients, = torch.autograd.grad(logits.gather(1, labels[:, None]).sum(), activation)
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        cam = F.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return cam if signed else _normalize(cam)


@torch.enable_grad()
def integrated_gradients(model: nn.Module, images: torch.Tensor,
                         target: torch.Tensor | None = None, baseline: torch.Tensor | None = None,
                         steps: int = 32, forward_batch_size: int = 64, *, signed=False) -> torch.Tensor:
    baseline = torch.zeros_like(images) if baseline is None else baseline
    with torch.no_grad():
        labels = model(images).argmax(1) if target is None else target
    delta = images - baseline
    alphas = torch.linspace(0.0, 1.0, steps, device=images.device, dtype=images.dtype)
    weights = torch.ones_like(alphas)
    weights[[0, -1]] = 0.5
    total_gradients = torch.zeros_like(images)
    for alpha, weight in zip(alphas.split(max(1, forward_batch_size // len(images))),
                             weights.split(max(1, forward_batch_size // len(images)))):
        scaled = (baseline + alpha[:, None, None, None, None] * delta).detach().requires_grad_(True)
        logits = model(scaled.flatten(0, 1))
        scores = logits.gather(1, labels.repeat(len(alpha))[:, None])
        gradients, = torch.autograd.grad(scores.sum(), scaled)
        total_gradients += (gradients * weight[:, None, None, None, None]).sum(0)
    attribution = delta * total_gradients / (steps - 1)
    if signed:
        return attribution.sum(dim=1, keepdim=True)
    return _normalize(attribution.abs().sum(dim=1, keepdim=True))


@torch.no_grad()
def occlusion_sensitivity(model: nn.Module, images: torch.Tensor,
                          patch_size: int = 32, stride: int = 16,
                          fill_value: float | torch.Tensor = 0.0, target: torch.Tensor | None = None,
                          forward_batch_size: int = 64, *, signed=False) -> torch.Tensor:
    baseline_scores = model(images).softmax(1)
    labels = baseline_scores.argmax(1) if target is None else target
    baseline = baseline_scores.gather(1, labels[:, None]).squeeze(1)
    height, width = images.shape[-2:]
    maps = images.new_zeros((len(images), 1, height, width))
    counts = images.new_zeros((height, width))
    patches = [(slice(y, min(y + patch_size, height)), slice(x, min(x + patch_size, width)))
               for y in range(0, height, stride) for x in range(0, width, stride)]
    chunk_size = max(1, forward_batch_size // len(images))
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start:start + chunk_size]
        occluded = images.expand(len(chunk), *images.shape).clone()
        for position, (y, x) in enumerate(chunk):
            occluded[position, :, :, y, x] = fill_value
        scores = model(occluded.flatten(0, 1)).softmax(1)
        drops = baseline - scores.gather(1, labels.repeat(len(chunk))[:, None]).reshape(len(chunk), len(images))
        for drop, (y, x) in zip(drops, chunk):
            maps[:, 0, y, x] += drop[:, None, None]
            counts[y, x] += 1
    maps = maps / counts.clamp_min(1)
    return maps if signed else _normalize(maps.abs())
