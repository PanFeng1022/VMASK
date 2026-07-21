from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantizationResult:
    q: object
    adjusted_coordinates: int
    total_coordinates: int


def stochastic_quantize(update, psi: float, amax: int, generator) -> QuantizationResult:
    import torch

    scaled_raw = update.detach().cpu() * float(psi)
    finite = torch.isfinite(scaled_raw)
    adjusted = ((scaled_raw.abs() > float(amax)) | (~finite)).sum().item()
    scaled = scaled_raw.clone()
    scaled = torch.where(torch.isnan(scaled), torch.zeros_like(scaled), scaled)
    scaled = torch.where(torch.isinf(scaled) & (scaled > 0), torch.full_like(scaled, float(amax)), scaled)
    scaled = torch.where(torch.isinf(scaled) & (scaled < 0), torch.full_like(scaled, -float(amax)), scaled)
    clipped = torch.clamp(scaled, min=-float(amax), max=float(amax))
    lower = torch.floor(clipped)
    prob_up = clipped - lower
    draws = torch.rand(prob_up.shape, generator=generator)
    q = (lower + (draws < prob_up).to(lower.dtype)).to(torch.int64)
    return QuantizationResult(
        q=q,
        adjusted_coordinates=int(adjusted),
        total_coordinates=int(q.numel()),
    )


def check_modulus(modulus: int, num_clients: int, amax: int) -> None:
    minimum = 2 * num_clients * amax
    if modulus <= minimum:
        raise ValueError(f"modulus must be > {minimum}; got {modulus}")


def signed_mod_decode(values, modulus: int):
    import torch

    encoded = torch.remainder(values, modulus)
    half = modulus // 2
    return torch.where(encoded >= half, encoded - modulus, encoded)


def aggregate_update_from_qsum(q_sum, num_clients: int, psi: float, device):
    import torch

    return q_sum.to(dtype=torch.float32, device=device) / (float(num_clients) * float(psi))


def relative_eq_error(left, right, eps: float = 1e-12) -> float:
    import torch

    numerator = torch.norm((left - right).to(dtype=torch.float64), p=2)
    denominator = torch.norm(left.to(dtype=torch.float64), p=2) + eps
    return float((numerator / denominator).item())
