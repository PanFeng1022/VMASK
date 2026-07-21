from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import pickle
import struct

import numpy as np


@dataclass
class PartitionResult:
    client_indices: list[list[int]]
    label_counts: np.ndarray


def load_mnist(data_dir: str | Path, download: bool):
    local = _load_local_mnist(data_dir)
    if local is not None:
        return local

    original_data_dir = Path(__file__).resolve().parents[4] / "原稿的实验" / "data"
    local = _load_local_mnist(original_data_dir)
    if local is not None:
        return local

    from torchvision import datasets, transforms

    _patch_mnist_download_mirror(datasets)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )
    return train, test


def load_dataset(name: str, data_dir: str | Path, download: bool):
    normalized = name.lower().replace("_", "-")
    if normalized == "mnist":
        return load_mnist(data_dir, download)
    if normalized in {"cifar10", "cifar-10"}:
        return load_cifar10(data_dir, download)
    raise ValueError(f"Unknown dataset: {name}")


def load_cifar10(data_dir: str | Path, download: bool):
    local = _load_local_cifar10(data_dir)
    if local is not None:
        return local

    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )
    return train, test


def _load_local_cifar10(data_dir: str | Path):
    root = Path(data_dir) / "cifar-10-batches-py"
    train_files = [root / f"data_batch_{idx}" for idx in range(1, 6)]
    test_file = root / "test_batch"
    if not all(path.exists() for path in train_files) or not test_file.exists():
        return None
    return (
        LocalCIFAR10(train_files),
        LocalCIFAR10([test_file]),
    )


def _load_local_mnist(data_dir: str | Path):
    raw = Path(data_dir) / "MNIST" / "raw"
    train_images = _find_mnist_file(raw, "train-images-idx3-ubyte")
    train_labels = _find_mnist_file(raw, "train-labels-idx1-ubyte")
    test_images = _find_mnist_file(raw, "t10k-images-idx3-ubyte")
    test_labels = _find_mnist_file(raw, "t10k-labels-idx1-ubyte")
    if not all([train_images, train_labels, test_images, test_labels]):
        return None

    return (
        LocalMNIST(train_images, train_labels),
        LocalMNIST(test_images, test_labels),
    )


def _find_mnist_file(raw_dir: Path, stem: str) -> Path | None:
    plain = raw_dir / stem
    if plain.exists():
        return plain
    gz = raw_dir / f"{stem}.gz"
    if gz.exists():
        return gz
    return None


def _read_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _read_idx_images(path: Path) -> np.ndarray:
    data = _read_bytes(path)
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected MNIST image magic {magic} in {path}")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    data = _read_bytes(path)
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected MNIST label magic {magic} in {path}")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if labels.shape[0] != count:
        raise ValueError(f"MNIST label count mismatch in {path}")
    return labels.astype(np.int64)


class LocalMNIST:
    def __init__(self, image_path: Path, label_path: Path):
        import torch

        images = _read_idx_images(image_path)
        labels = _read_idx_labels(label_path)
        if images.shape[0] != labels.shape[0]:
            raise ValueError("MNIST image/label count mismatch")

        tensor = torch.from_numpy(images.copy()).to(dtype=torch.float32).unsqueeze(1) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        self.data = tensor
        self.targets = torch.from_numpy(labels.copy()).to(dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


class LocalCIFAR10:
    def __init__(self, batch_paths: list[Path]):
        import torch

        arrays = []
        labels = []
        for path in batch_paths:
            with path.open("rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            arrays.append(batch["data"])
            labels.extend(batch.get("labels", batch.get("fine_labels")))

        data = np.concatenate(arrays, axis=0).reshape(-1, 3, 32, 32)
        tensor = torch.from_numpy(data.copy()).to(dtype=torch.float32) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)
        self.data = (tensor - mean) / std
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def _patch_mnist_download_mirror(datasets) -> None:
    # Older torchvision releases still point MNIST downloads to the original
    # Yann LeCun HTTP host, which often returns 404. Keep the official file
    # hashes but use PyTorch's maintained dataset mirror.
    datasets.MNIST.resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]


def dataset_labels(dataset) -> np.ndarray:
    targets = dataset.targets
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


def partition_dataset(
    labels: np.ndarray,
    num_clients: int,
    partition: str,
    seed: int,
    alpha: float = 0.5,
) -> PartitionResult:
    if len(labels) % num_clients != 0:
        raise ValueError("Balanced partitioning requires dataset size divisible by clients")

    rng = np.random.default_rng(seed)
    if partition == "iid":
        client_indices = _iid_partition(labels, num_clients, rng)
    elif partition == "dirichlet":
        client_indices = _balanced_dirichlet_partition(labels, num_clients, alpha, rng)
    else:
        raise ValueError(f"Unknown partition: {partition}")

    counts = label_distribution(labels, client_indices)
    return PartitionResult(client_indices=client_indices, label_counts=counts)


def _iid_partition(labels: np.ndarray, num_clients: int, rng: np.random.Generator):
    indices = rng.permutation(len(labels))
    per_client = len(labels) // num_clients
    return [indices[i * per_client : (i + 1) * per_client].tolist() for i in range(num_clients)]


def _largest_remainder_counts(weights: np.ndarray, total: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    return counts


def _balanced_dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
):
    num_classes = int(labels.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = _largest_remainder_counts(proportions, len(class_indices))
        start = 0
        for client_id, count in enumerate(counts):
            stop = start + int(count)
            client_indices[client_id].extend(class_indices[start:stop].tolist())
            start = stop

    _rebalance_client_sizes(client_indices, len(labels) // num_clients, rng)
    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def _rebalance_client_sizes(
    client_indices: list[list[int]],
    target_size: int,
    rng: np.random.Generator,
) -> None:
    while True:
        over = [i for i, idx in enumerate(client_indices) if len(idx) > target_size]
        under = [i for i, idx in enumerate(client_indices) if len(idx) < target_size]
        if not over and not under:
            return
        if not over or not under:
            raise RuntimeError("Could not rebalance client partitions")

        src = max(over, key=lambda i: len(client_indices[i]))
        dst = min(under, key=lambda i: len(client_indices[i]))
        move_count = min(len(client_indices[src]) - target_size, target_size - len(client_indices[dst]))

        positions = rng.choice(len(client_indices[src]), size=move_count, replace=False)
        positions = np.sort(positions)[::-1]
        moved = []
        for pos in positions:
            moved.append(client_indices[src].pop(int(pos)))
        client_indices[dst].extend(moved)


def label_distribution(labels: np.ndarray, client_indices: list[list[int]]) -> np.ndarray:
    num_classes = int(labels.max()) + 1
    counts = np.zeros((len(client_indices), num_classes), dtype=np.int64)
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts[client_id] = np.bincount(client_labels, minlength=num_classes)
    return counts


def save_label_distribution(path: str | Path, counts: np.ndarray) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "total"] + [f"label_{i}" for i in range(counts.shape[1])])
        for client_id, row in enumerate(counts):
            writer.writerow([client_id, int(row.sum()), *[int(v) for v in row]])
