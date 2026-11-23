import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=64):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("VAE image size must be a positive multiple of 16.")
        self.latent_dim = latent_dim
        self.side = image_size // 16
        channels = (3, 32, 64, 96, 128)
        self.encoder = nn.Sequential(*[
            layer for incoming, outgoing in zip(channels, channels[1:])
            for layer in (nn.Conv2d(incoming, outgoing, 4, 2, 1), nn.SiLU())
        ])
        features = channels[-1] * self.side**2
        self.moments = nn.Linear(features, 2 * latent_dim)
        self.expand = nn.Linear(latent_dim, features)
        self.decoder = nn.Sequential(*[
            layer for incoming, outgoing in zip(reversed(channels[1:]), reversed(channels[:-1]))
            for layer in (nn.ConvTranspose2d(incoming, outgoing, 4, 2, 1),
                          nn.Sigmoid() if outgoing == 3 else nn.SiLU())
        ])

    def encode(self, images):
        mu, log_variance = self.moments(self.encoder(images).flatten(1)).chunk(2, dim=1)
        return mu, log_variance.clamp(-20, 10)

    def decode(self, latent):
        expanded = self.expand(latent).reshape(len(latent), 128, self.side, self.side)
        return self.decoder(expanded)

    def forward(self, images, sample=True):
        mu, log_variance = self.encode(images)
        latent = mu + torch.randn_like(mu) * (0.5 * log_variance).exp() if sample else mu
        return self.decode(latent), mu, log_variance


def vae_loss(images, reconstruction, mu, log_variance, beta):
    mse = F.mse_loss(reconstruction, images)
    kl = -0.5 * (1 + log_variance - mu.square() - log_variance.exp()).sum(dim=1).mean()
    return mse + beta * kl, mse, kl
