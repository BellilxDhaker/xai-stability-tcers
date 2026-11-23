import pandas as pd
import torch

from src.config import config_from_args
from src.data.oxford_pets import dataset_tables, load_oxford_pets
from src.evaluation.common import setup
from src.models.resnet import PixelClassifier, build_resnet18, evaluate_classifier, train_model
from src.utils import save_json
from src.visualization.plots import plot_training_history


def run(config):
    target_device = setup(config, "classifier")
    torch.hub.set_dir(str(config.checkpoint_root / "torch_hub"))
    limits = (config.quick_train_samples, config.quick_validation_samples, config.test_samples) if config.quick else None
    train_loader, validation_loader, test_loader = load_oxford_pets(
        config.data_root, config.image_size, config.batch_size, config.num_workers,
        validation_fraction=config.validation_fraction, seed=config.seed, limits=limits,
    )
    model = PixelClassifier(build_resnet18(
        config.num_classes, pretrained=config.pretrained and config.resnet_checkpoint is None,
    ))
    if config.resnet_checkpoint is None:
        history = train_model(model, train_loader, validation_loader, target_device, config.epochs,
                              config.learning_rate, config.weight_decay, config.patience, config.classifier_path)
        history = pd.DataFrame(history)
        history.to_csv(config.output_root / "training_history.csv", index=False)
        plot_training_history(history, config.figures / "training_history.png")
    checkpoint = torch.load(config.classifier_path, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_classifier(model.to(target_device), test_loader, target_device)
    metrics["n"] = len(test_loader.dataset)
    save_json(metrics, config.output_root / "test_metrics.json")
    dataset_tables(config).to_csv(config.output_root / "dataset_splits.csv", index=False)
    print(metrics, flush=True)


def main():
    run(config_from_args("Train and evaluate the Oxford-IIIT Pet classifier."))


if __name__ == "__main__":
    main()
