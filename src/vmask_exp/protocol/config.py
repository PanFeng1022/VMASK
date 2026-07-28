from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class ProtocolConfig:
    modulus: int = 2**128
    field_modulus: int = (1 << 192) - (1 << 64) - 1
    amax: int = 15
    r0: int = 1023
    limbs: int = 10
    beta: int = 2047
    committee_size: int = 5
    threshold: int = 3

    @property
    def mask_bound(self) -> int:
        return (self.beta**self.limbs - 1) // 2

    def validate(self, dimension: int, max_clients: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not (2 <= self.threshold <= self.committee_size):
            raise ValueError("threshold must satisfy 2 <= threshold <= committee_size")
        if self.beta != 2 * self.r0 + 1:
            raise ValueError("beta must equal 2*r0+1")
        mask_bound = self.mask_bound
        if self.amax + mask_bound >= self.modulus:
            raise ValueError("modulus is too small for the mask bound")
        if self.modulus <= 2 * max_clients * self.amax:
            raise ValueError("modulus is too small for aggregate decoding")
        b_a = math.ceil(math.log2(2 * self.amax + 1))
        b_s = math.ceil(math.log2(2 * self.r0 + 1))
        field_required = max(
            self.modulus + self.amax + mask_bound,
            2 * max_clients * mask_bound,
            2 * ((1 << b_a) - 1),
            2 * ((1 << b_s) - 1),
        )
        if self.field_modulus <= field_required:
            raise ValueError("field_modulus is too small for direct mask recovery")


def protocol_parameter_digest(
    cfg: ProtocolConfig,
    *,
    dimension: int,
    max_clients: int,
) -> str:
    payload = {
        "version": "VMASK-direct-mask-v1",
        "dimension": int(dimension),
        "max_clients": int(max_clients),
        "modulus": str(cfg.modulus),
        "field_modulus": str(cfg.field_modulus),
        "amax": cfg.amax,
        "r0": cfg.r0,
        "limbs": cfg.limbs,
        "beta": cfg.beta,
        "committee_size": cfg.committee_size,
        "threshold": cfg.threshold,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def profile_p80(
    *,
    modulus: int = 2**128,
    committee_size: int = 5,
    threshold: int = 3,
) -> ProtocolConfig:
    return ProtocolConfig(
        modulus=modulus,
        r0=1023,
        limbs=10,
        beta=2047,
        committee_size=committee_size,
        threshold=threshold,
    )


def profile_p100(
    *,
    modulus: int = 2**128,
    committee_size: int = 5,
    threshold: int = 3,
) -> ProtocolConfig:
    return ProtocolConfig(
        modulus=modulus,
        r0=8191,
        limbs=9,
        beta=16383,
        committee_size=committee_size,
        threshold=threshold,
    )
