import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import ExperimentConfig
from src.data.oxford_pets import load_oxford_pets
from src.models.resnet import build_resnet18, evaluate_classifier, train_model
from src.utils import device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Oxford-IIIT Pet ResNet18 baseline.")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    config = ExperimentConfig(epochs=args.epochs)
    config.create_directories()
    seed_everything(config.seed)
    train_loader, validation_loader, test_loader = load_oxford_pets(
        config.data_root, config.image_size, config.batch_size, config.num_workers, seed=config.seed)
    model = build_resnet18(config.num_classes)
    history = train_model(model, train_loader, validation_loader,
                          device(), config.epochs, config.learning_rate, config.weight_decay,
                          config.patience, config.checkpoint_root / "resnet18_best.pth")
    pd.DataFrame(history).to_csv(config.output_root / "training_history.csv", index=False)
    checkpoint = config.checkpoint_root / "resnet18_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device())["model_state_dict"])
    test_metrics = evaluate_classifier(model, test_loader, device())
    save_json(test_metrics, config.output_root / "test_metrics.json")
    print("Test metrics:", test_metrics, flush=True)
    save_json({
        "seed": config.seed,
        "epochs": config.epochs,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "patience": config.patience,
        "num_classes": config.num_classes,
    }, config.output_root / "training_config.json")
    print(f"Saved training artifacts to {config.output_root}", flush=True)


if __name__ == "__main__":
    main()
