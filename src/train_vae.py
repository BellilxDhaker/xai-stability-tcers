import math

import pandas as pd
import torch
from torchvision.utils import save_image

from src.config import config_from_args
from src.data.oxford_pets import make_loader, train_validation
from src.evaluation.common import setup
from src.models.vae import ConvVAE, vae_loss
from src.perturbations.vae_perturbations import vae_checkpoint
from src.utils import save_checkpoint, seed_everything


def epoch_pass(model, loader, target_device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = [0.0, 0.0, 0.0]
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            images = images.to(target_device)
            reconstruction, mu, log_variance = model(images, sample=training)
            losses = vae_loss(images, reconstruction, mu, log_variance, beta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                optimizer.step()
            totals = [total + loss.item() * len(images) for total, loss in zip(totals, losses)]
    return {name: total / len(loader.dataset) for name, total in zip(("loss", "mse", "kl"), totals)}


def run(config):
    target_device = setup(config, "vae")
    train, validation = train_validation(config.data_root, config.vae_size, config.validation_fraction,
                                         config.seed, augment=False)
    if config.quick:
        train.indices = train.indices[:config.quick_train_samples]
        validation.indices = validation.indices[:config.quick_validation_samples]
    validation_loader = make_loader(validation, config.batch_size, config.num_workers)
    rows = []
    for seed in config.evaluation_seeds:
        seed_everything(seed)
        train_loader = make_loader(train, config.batch_size, config.num_workers, shuffle=True, seed=seed)
        model = ConvVAE(config.vae_size, config.latent_dim).to(target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.vae_learning_rate)
        best = math.inf
        path = vae_checkpoint(config, seed)
        for epoch in range(1, config.vae_epochs + 1):
            row = {"seed": seed, "epoch": epoch}
            for name, loader, opt in (("train", train_loader, optimizer), ("validation", validation_loader, None)):
                row.update({f"{name}_{key}": value for key, value in
                            epoch_pass(model, loader, target_device, config.vae_beta, opt).items()})
            if row["validation_loss"] < best:
                best = row["validation_loss"]
                save_checkpoint({"model_state_dict": model.state_dict(), "epoch": epoch, "seed": seed,
                                 "image_size": config.vae_size, "latent_dim": config.latent_dim}, path)
            rows.append(row)
            print(row, flush=True)
        model.load_state_dict(torch.load(path, map_location=target_device, weights_only=True)["model_state_dict"])
        with torch.no_grad():
            images = next(iter(validation_loader))[0][:8].to(target_device)
            reconstructions, _, _ = model(images, sample=False)
            samples = model.decode(torch.randn(len(images), config.latent_dim, device=target_device))
            save_image(torch.cat((images, reconstructions, samples)), config.figures / f"vae_grid_{seed}.png",
                       nrow=len(images), padding=2)
    pd.DataFrame(rows).to_csv(config.output_root / "vae_history.csv", index=False)


def main():
    run(config_from_args("Train convolutional VAEs on the official training subset."))


if __name__ == "__main__":
    main()
