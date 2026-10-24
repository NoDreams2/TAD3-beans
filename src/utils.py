"""Общие функции: параметры, воспроизводимость, устройство."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
AUGMENTED_DIR = Path("data/augmented")


def load_params(path="params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел, чтобы результат повторялся."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
