from dataclasses import dataclass, field

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLASSIC_PARAMETERS = {
    "rotation": (-15, -10, -5, -2, 2, 5, 10, 15),
    "brightness": (0.8, 0.95, 0.99, 1.01, 1.05, 1.2),
    "noise": (0.002, 0.005, 0.01, 0.03, 0.05),
    "resize": (128, 180, 256),
}


@dataclass(frozen=True)
class Perturbation:
    name: str
    parameter: float
    noise_generator: torch.Generator | None = field(default=None, repr=False, compare=False)

    def _geometry(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.name == "rotation":
            return TF.rotate(tensor, self.parameter, interpolation=InterpolationMode.BILINEAR, fill=None)
        if self.name == "resize":
            size = int(self.parameter)
            resized = TF.resize(tensor, [size, size], antialias=True)
            return TF.resize(resized, list(tensor.shape[-2:]), antialias=True)
        return tensor

    def apply(self, image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.name == "brightness":
            return (image * self.parameter).clamp(0, 1)
        if self.name == "noise":
            generator = self.noise_generator if generator is None else generator
            noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
            return (image + noise * self.parameter).clamp(0, 1)
        return self._geometry(image).clamp(0, 1)

    def align(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        aligned = self._geometry(maps)
        if self.name != "rotation" or self.parameter == 0:
            return aligned, None
        height, width = maps.shape[-2:]
        support = self._geometry(maps.new_ones((1, 1, height, width)))[0, 0]
        return aligned, support >= 1.0 - 1e-6


def make_perturbations(noise_generator=None, *, parameters=None):
    parameters = CLASSIC_PARAMETERS if parameters is None else parameters
    if set(parameters) - CLASSIC_PARAMETERS.keys():
        raise ValueError(f"Unknown perturbations: {set(parameters) - CLASSIC_PARAMETERS.keys()}")
    return [Perturbation(name, value, noise_generator) for name, values in parameters.items() for value in values]
