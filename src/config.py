from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class ExperimentConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("experiments/results")
    checkpoint_root: Path = Path("checkpoints")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_classes: int = 2
    perturbations: Dict[str, Tuple[float, ...]] = field(default_factory=lambda: {
        "rotation": (-15.0, -10.0, -5.0, 5.0, 10.0, 15.0),
        "brightness": (0.8, 1.2),
        "noise": (0.01, 0.03, 0.05),
        "resize": (128.0, 180.0, 256.0),
    })

    def create_directories(self) -> None:
        for path in (self.data_root, self.output_root, self.checkpoint_root,
                     Path("figures")):
            path.mkdir(parents=True, exist_ok=True)

