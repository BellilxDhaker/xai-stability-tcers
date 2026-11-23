import hashlib
import os
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from src.explainers.methods import GradCAM, integrated_gradients, occlusion_sensitivity
from src.models.resnet import PixelClassifier, build_resnet18
from src.utils import device, seed_everything

METHODS = ("grad_cam", "integrated_gradients", "occlusion", "random", "constant")
LEARNED_METHODS = METHODS[:3]


def setup(config, stage):
    config.create_directories()
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    config.save(stage, device=str(device()), torch_version=str(torch.__version__))
    return device()


def classifier(config, target_device):
    checkpoint = config.classifier_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint missing: {checkpoint}. Run python -m src.train first.")
    network = build_resnet18(config.num_classes, pretrained=False)
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(contents["model_state_dict"])
    with checkpoint.open("rb") as handle:
        checkpoint_key = hashlib.file_digest(handle, "sha256").hexdigest()
    return PixelClassifier(network).to(target_device).eval(), checkpoint_key


def stage_signature(config, checkpoint_key):
    return {
        "classifier_sha256": checkpoint_key, "cohort_seed": config.seed, "cohort_size": config.test_samples,
        "evaluation_seeds": list(config.evaluation_seeds), "image_size": config.image_size,
        "evaluation_batch_size": config.evaluation_batch_size, "num_classes": config.num_classes,
        "ig_steps": config.ig_steps, "occlusion_patch": config.occlusion_patch,
        "occlusion_stride": config.occlusion_stride, "metric_size": config.metric_size,
        "vae_size": config.vae_size, "latent_dim": config.latent_dim,
    }


def explainers(model, config):
    return {
        "grad_cam": GradCAM(model, model.backbone.layer4[-1]),
        "integrated_gradients": partial(integrated_gradients, model, steps=config.ig_steps,
                                       forward_batch_size=config.forward_batch_size),
        "occlusion": partial(occlusion_sensitivity, model, patch_size=config.occlusion_patch,
                             stride=config.occlusion_stride, fill_value=model.mean[0],
                             forward_batch_size=config.forward_batch_size),
    }


def make_generator(target_device, *labels):
    digest = hashlib.sha256(repr(labels).encode()).digest()
    return torch.Generator(device=target_device).manual_seed(int.from_bytes(digest[:8], "little"))


class ExplanationMaps:
    def __init__(self, model, config, checkpoint_key):
        self.explainers = explainers(model, config)
        self.cache = config.output_root / "cache" if config.cache_maps else None
        method_source = Path(__file__).parent.parent / "explainers" / "methods.py"
        self.key = (checkpoint_key, hashlib.sha256(method_source.read_bytes()).hexdigest(),
                    config.ig_steps, config.occlusion_patch, config.occlusion_stride, config.forward_batch_size,
                    torch.__version__, str(model.mean.device))

    def __call__(self, images, targets, *, signed=False, random_seed=0):
        pixels = images.detach().cpu().contiguous()
        cache_path = None
        if self.cache is not None:
            digest = hashlib.sha256(repr((self.key, tuple(pixels.shape), signed)).encode())
            digest.update(pixels.numpy().tobytes())
            digest.update(targets.cpu().numpy().tobytes())
            key = digest.hexdigest()
            cache_path = self.cache / key[:2] / f"{key}.npy"

        try:
            values = np.load(cache_path, allow_pickle=False) if cache_path is not None else None
        except (FileNotFoundError, OSError, ValueError):
            values = None
        if values is None:
            values = np.stack([method(images, targets, signed=signed).cpu().numpy()
                               for method in self.explainers])
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    np.save(handle, values, allow_pickle=False)
                try:
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)

        result = {name: torch.from_numpy(values[i]) for i, name in enumerate(self.explainers)}
        shape = result["grad_cam"].shape
        draw = torch.randn if signed else torch.rand
        result["random"] = draw(shape, generator=make_generator("cpu", random_seed))
        result["constant"] = torch.ones(shape)
        return result
