from dataclasses import dataclass
from typing import Callable, List

import torch
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Perturbation:
    name: str
    parameter: float
    apply: Callable[[torch.Tensor], torch.Tensor]


def make_perturbations(noise_generator: torch.Generator | None = None) -> List[Perturbation]:
    def rotation(angle: float):
        return lambda image: TF.rotate(image, angle)

    def brightness(factor: float):
        return lambda image: TF.adjust_brightness(image, factor)

    def noise(std: float):
        def apply(image):
            noise_tensor = torch.randn(image.shape, generator=noise_generator, device=image.device) * std
            return image + noise_tensor
        return apply

    def resize(size: float):
        return lambda image: TF.resize(image, [int(size), int(size)])

    return ([Perturbation("rotation", x, rotation(x)) for x in (-15, -10, -5, 5, 10, 15)] +
            [Perturbation("brightness", x, brightness(x)) for x in (0.8, 1.2)] +
            [Perturbation("noise", x, noise(x)) for x in (0.01, 0.03, 0.05)] +
            [Perturbation("resize", x, resize(x)) for x in (128, 180, 256)])

