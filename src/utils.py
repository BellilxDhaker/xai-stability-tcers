import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_checkpoint(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        staging_path = Path(temporary.name)
    try:
        torch.save(payload, staging_path)
        staging_path.replace(path)
    finally:
        staging_path.unlink(missing_ok=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

