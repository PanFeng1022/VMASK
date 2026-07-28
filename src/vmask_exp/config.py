from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    experiment_name: str = "benign_mnist"
    data_dir: str = "data"
    dataset: str = "mnist"
    model: str = "mnist_cnn"
    results_dir: str = "results/benign_mnist"
    download: bool = False
    device: str = "auto"
    overwrite: bool = False
    skip_completed: bool = False

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    partitions: list[str] = field(default_factory=lambda: ["iid", "dirichlet"])
    dirichlet_alpha: float = 0.5

    num_clients: int = 20
    participating_clients: int = 20
    rounds: int = 300
    local_steps: int = 20
    batch_size: int = 64
    local_lr: float = 0.05
    momentum: float = 0.9
    server_lr: float = 0.1
    optimizer: str = "sgd"

    psi: float = 256.0
    amax: int = 15
    modulus: int = 2**128
    expected_dimension: int = 27_562

    protocol_execution: bool = True
    protocol_profile: str = "p80"
    protocol_committee_size: int = 5
    protocol_threshold: int = 3
    protocol_modulus: int = 2**128

    eval_batch_size: int = 512
    num_workers: int = 0
    final_window: int = 10


def load_config(path: str | Path | None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg

    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    unknown = sorted(set(data) - set(asdict(cfg)))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    for key, value in data.items():
        setattr(cfg, key, value)
    return cfg


def _load_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("PyYAML is not installed and the config uses unsupported YAML syntax")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() == "true":
            parsed: Any = True
        elif value.lower() == "false":
            parsed = False
        elif value.lower() in {"null", "none", ""}:
            parsed = None
        elif value.startswith("[") and value.endswith("]"):
            parsed = _parse_simple_list(value)
        else:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value
        data[key] = parsed
    return data


def _parse_simple_list(value: str) -> list[Any]:
    inner = value[1:-1].strip()
    if not inner:
        return []
    items: list[Any] = []
    for part in inner.split(","):
        token = part.strip()
        try:
            items.append(ast.literal_eval(token))
        except (ValueError, SyntaxError):
            items.append(token.strip("'\""))
    return items


def config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
