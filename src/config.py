import argparse
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from src.perturbations.transforms import CLASSIC_PARAMETERS
from src.utils import save_json


@dataclass
class ExperimentConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("experiments/project4")
    checkpoint_root: Path = Path("checkpoints")
    resnet_checkpoint: Path | None = None
    image_size: int = 224
    batch_size: int = 32
    evaluation_batch_size: int = 8
    forward_batch_size: int = 64
    num_workers: int = 0
    cpu_threads: int = 4
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    evaluation_seeds: tuple[int, ...] = (42, 43, 44)
    test_samples: int = 300
    num_classes: int = 2
    validation_fraction: float = 0.15
    pretrained: bool = True
    ig_steps: int = 16
    occlusion_patch: int = 56
    occlusion_stride: int = 56
    metric_size: int = 64
    faithfulness_steps: int = 20
    faithfulness_baselines: tuple[str, ...] = ("blur", "mean")
    bootstrap_repeats: int = 2000
    vae_size: int = 128
    latent_dim: int = 64
    vae_epochs: int = 20
    vae_learning_rate: float = 1e-3
    vae_beta: float = 0.001
    latent_sigmas: tuple[float, ...] = (0.25, 0.5, 1.0)
    match_caliper: float = 0.003
    cache_maps: bool = True
    quick: bool = False
    quick_train_samples: int = 128
    quick_validation_samples: int = 32
    quick_test_samples: int = 8
    perturbations: dict[str, tuple[float, ...]] = field(default_factory=CLASSIC_PARAMETERS.copy)

    def __post_init__(self):
        for name in ("data_root", "output_root", "checkpoint_root", "resnet_checkpoint"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))
        if not self.quick and (self.test_samples < 300 or len(set(self.evaluation_seeds)) < 3):
            raise ValueError("A full run requires at least 300 test images and three distinct evaluation seeds.")
        if self.occlusion_stride > self.occlusion_patch or self.occlusion_stride < 1:
            raise ValueError("Occlusion stride must be positive and no larger than the patch.")
        if self.metric_size < 32 or self.ig_steps < 2 or self.faithfulness_steps < 2:
            raise ValueError("Use metric_size >= 32, ig_steps >= 2 and faithfulness_steps >= 2.")
        if (not self.faithfulness_baselines or not self.latent_sigmas or self.match_caliper < 0
                or any(sigma <= 0 for sigma in self.latent_sigmas) or not any(self.perturbations.values())):
            raise ValueError("Specify faithfulness baselines, latent sigmas and a nonnegative caliper.")

    @property
    def classifier_path(self) -> Path:
        return self.resnet_checkpoint or self.checkpoint_root / "resnet18_best.pth"

    @property
    def figures(self) -> Path:
        return self.output_root / "figures"

    def create_directories(self) -> None:
        for path in (self.data_root, self.output_root, self.checkpoint_root, self.figures):
            path.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, **details) -> None:
        save_json({**asdict(self), **details}, self.output_root / f"config_{stage}.json")


def config_from_args(description: str, argv=None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="JSON overrides for ExperimentConfig")
    parser.add_argument("--quick", action="store_true", default=None, help="Small smoke run")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resnet-checkpoint", type=Path)
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    known = {field.name for field in fields(ExperimentConfig)}
    unknown = set(values) - known
    if unknown:
        parser.error(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    values.update({name: value for name, value in vars(args).items() if name != "config" and value is not None})
    config = ExperimentConfig(**values)
    if config.quick:
        quick_defaults = {
            "output_root": Path("experiments/project4-quick"),
            "checkpoint_root": Path("checkpoints/quick"),
            "test_samples": config.quick_test_samples,
            "epochs": 1,
            "vae_epochs": 1,
            "pretrained": False,
            "image_size": 64,
            "vae_size": 32,
            "metric_size": 32,
            "ig_steps": 4,
            "occlusion_patch": 32,
            "occlusion_stride": 32,
            "faithfulness_steps": 4,
            "evaluation_seeds": (config.seed,),
            "bootstrap_repeats": 100,
            "batch_size": 16,
            "perturbations": {"rotation": (10,), "brightness": (1.01,), "noise": (0.0002, 0.001, 0.01), "resize": (48,)},
        }
        config = replace(config, **{key: value for key, value in quick_defaults.items() if key not in values})
    return config
