from __future__ import annotations


def create_mnist_cnn():
    import torch.nn as nn
    import torch.nn.functional as F

    class LightweightMNISTCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(16 * 5 * 5, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return LightweightMNISTCNN()


def create_cifar10_cnn():
    import torch.nn as nn
    import torch.nn.functional as F

    class CompactCIFAR10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return CompactCIFAR10CNN()


def create_model(name: str):
    normalized = name.lower().replace("_", "-")
    if normalized in {"mnist", "mnist-cnn", "lightweight-mnist-cnn"}:
        return create_mnist_cnn()
    if normalized in {"cifar10", "cifar-10", "cifar10-cnn", "cifar-10-cnn", "compact-cifar10-cnn"}:
        return create_cifar10_cnn()
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_parameter_count(model, expected: int) -> None:
    actual = parameter_count(model)
    if actual != expected:
        raise ValueError(f"Model dimension mismatch: expected {expected}, got {actual}")
