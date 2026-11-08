from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def _normalize(attribution: torch.Tensor) -> torch.Tensor:
    attribution = attribution.abs().sum(dim=1, keepdim=True)
    minimum = attribution.amin(dim=(-2, -1), keepdim=True)
    maximum = attribution.amax(dim=(-2, -1), keepdim=True)
    return (attribution - minimum) / (maximum - minimum).clamp_min(1e-8)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self.target_layer = model, target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _, __, output):
        self.activations = output

    def _backward_hook(self, _, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, images: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        labels = logits.argmax(1) if target is None else target
        logits.gather(1, labels[:, None]).sum().backward()
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return _normalize(cam)


def integrated_gradients(model: nn.Module, images: torch.Tensor,
                         target: torch.Tensor | None = None,
                         baseline: torch.Tensor | None = None,
                         steps: int = 32) -> torch.Tensor:
    baseline = torch.zeros_like(images) if baseline is None else baseline
    with torch.no_grad():
        labels = model(images).argmax(1) if target is None else target
    total_gradients = torch.zeros_like(images)
    for alpha in torch.linspace(0.0, 1.0, steps, device=images.device):
        scaled = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
        logits = model(scaled)
        score = logits.gather(1, labels[:, None]).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        if scaled.grad is None:
            raise RuntimeError("Integrated gradients could not obtain input gradients.")
        total_gradients += scaled.grad.detach()
    attribution = (images - baseline) * total_gradients / steps
    return _normalize(attribution)


@torch.no_grad()
def occlusion_sensitivity(model: nn.Module, images: torch.Tensor,
                          patch_size: int = 32, stride: int = 16,
                          fill_value: float = 0.0) -> torch.Tensor:
    model.eval()
    baseline_scores = model(images).softmax(1)
    labels = baseline_scores.argmax(1)
    maps = torch.zeros((images.size(0), 1, images.size(2), images.size(3)), device=images.device)
    counts = torch.zeros_like(maps)
    for top in range(0, images.size(2), stride):
        for left in range(0, images.size(3), stride):
            bottom, right = min(top + patch_size, images.size(2)), min(left + patch_size, images.size(3))
            occluded = images.clone()
            occluded[:, :, top:bottom, left:right] = fill_value
            scores = model(occluded).softmax(1)
            drop = (baseline_scores[torch.arange(images.size(0)), labels] -
                    scores[torch.arange(images.size(0)), labels]).view(-1, 1, 1, 1)
            maps[:, :, top:bottom, left:right] += drop
            counts[:, :, top:bottom, left:right] += 1
    return _normalize(maps / counts.clamp_min(1))

