import hashlib

import torch
from torchvision.transforms import functional as TF

from src.models.vae import ConvVAE


def vae_checkpoint(config, seed):
    return config.checkpoint_root / f"vae_seed_{seed}.pth"


def vae_signatures(config):
    signatures = {}
    for seed in config.evaluation_seeds:
        with vae_checkpoint(config, seed).open("rb") as handle:
            signatures[seed] = hashlib.file_digest(handle, "sha256").hexdigest()
    return signatures


def load_vae(config, target_device, seed):
    path = vae_checkpoint(config, seed)
    if not path.is_file():
        raise FileNotFoundError(f"VAE checkpoint missing: {path}. Run python -m src.train_vae first.")
    contents = torch.load(path, map_location="cpu", weights_only=True)
    if (contents["image_size"], contents["latent_dim"], contents["seed"]) != (config.vae_size, config.latent_dim, seed):
        raise ValueError("The VAE checkpoint metadata differ from ExperimentConfig.")
    model = ConvVAE(contents["image_size"], contents["latent_dim"])
    model.load_state_dict(contents["model_state_dict"])
    return model.to(target_device).eval()


def resize_for_vae(images, image_size):
    return TF.resize(images, [image_size, image_size], antialias=True)


@torch.no_grad()
def decode_at_classifier_size(vae, latent, shape):
    return TF.resize(vae.decode(latent), list(shape), antialias=True).clamp(0, 1)


@torch.no_grad()
def latent_neighbourhood(vae, images, image_size, sigmas, generator):
    mu, _ = vae.encode(resize_for_vae(images, image_size))
    shape = images.shape[-2:]
    decoded = decode_at_classifier_size(vae, mu, shape)
    neighbours = [
        decode_at_classifier_size(vae, mu + sigma * torch.randn(
            mu.shape, generator=generator, device=mu.device, dtype=mu.dtype), shape)
        for sigma in sigmas
    ]
    return decoded, neighbours
