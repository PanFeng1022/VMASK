from __future__ import annotations

from dataclasses import dataclass

from .params import clone_model, parameter_vector


@dataclass
class ClientUpdate:
    update: object
    mean_loss: float


@dataclass
class EvalResult:
    loss: float
    accuracy: float


def train_client(
    global_model,
    dataset,
    indices: list[int],
    local_steps: int,
    batch_size: int,
    local_lr: float,
    momentum: float,
    device,
    seed: int,
    num_workers: int = 0,
) -> ClientUpdate:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    local_model = clone_model(global_model).to(device)
    local_model.train()

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
    )

    optimizer = torch.optim.SGD(local_model.parameters(), lr=local_lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    losses = []
    iterator = iter(loader)
    for _ in range(local_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = local_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    base_vec = parameter_vector(global_model).cpu()
    local_vec = parameter_vector(local_model).cpu()
    return ClientUpdate(update=local_vec - base_vec, mean_loss=sum(losses) / len(losses))


def evaluate_model(model, dataset, batch_size: int, device, num_workers: int = 0) -> EvalResult:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y).detach().cpu().item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach().cpu().item())
            total += int(y.numel())

    return EvalResult(loss=total_loss / total, accuracy=correct / total)
