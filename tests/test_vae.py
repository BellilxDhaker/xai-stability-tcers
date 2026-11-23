import pytest
import torch

from src.models.vae import ConvVAE, vae_loss


@pytest.mark.parametrize("size", [32, 128])
def test_vae_shapes_and_loss(size):
    torch.set_num_threads(2)
    model = ConvVAE(size, latent_dim=8)
    inputs = torch.rand(2, 3, size, size)
    reconstruction, mu, log_variance = model(inputs)
    assert reconstruction.shape == inputs.shape
    assert mu.shape == log_variance.shape == (2, 8)
    assert reconstruction.min() >= 0 and reconstruction.max() <= 1
    total, mse, kl = vae_loss(inputs, reconstruction, mu, log_variance, beta=0.001)
    assert torch.isfinite(total) and mse >= 0 and kl >= 0
    total.backward()
