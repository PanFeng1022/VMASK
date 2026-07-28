from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

from .config import ProtocolConfig


@dataclass(frozen=True)
class SnipStatement:
    theta_digest: str
    sid: str
    committee_ids: tuple[int, ...]
    client_id: int
    y: list[int]


@dataclass(frozen=True)
class SnipWitness:
    x: list[int]
    limbs: list[list[int]]
    shifted: list[list[int]]
    delta: list[int]
    z: list[int]


@dataclass(frozen=True)
class SnipConstraintResult:
    satisfied: bool
    verification_residual: int
    failed_constraints: int
    linear_constraints: int
    multiplication_constraints: int
    output: list[int]


@dataclass(frozen=True)
class SnipCommitteeMaterial:
    sigma_share: int
    z_shares: list[int]


@dataclass(frozen=True)
class SnipCheckResult:
    sigma_share: int
    z_shares: list[int]


@dataclass(frozen=True)
class SnipProofResult:
    relation_digest: str
    constraints: SnipConstraintResult
    committee_material: list[SnipCommitteeMaterial]


class VMaskSnip:

    def __init__(
        self,
        cfg: ProtocolConfig,
        dimension: int,
        *,
        theta_digest: str,
        committee_ids: Sequence[int],
        verification_ids: Sequence[int] | None = None,
    ):
        self.cfg = cfg
        self.dimension = int(dimension)
        self.output_count = self.dimension
        self.theta_digest = str(theta_digest)
        self.committee_ids = tuple(int(value) for value in committee_ids)
        if len(self.committee_ids) != cfg.committee_size:
            raise ValueError("committee identifier count does not match committee_size")
        if len(set(self.committee_ids)) != len(self.committee_ids):
            raise ValueError("committee identifiers must be unique")
        self.verification_ids = tuple(
            int(value)
            for value in (
                verification_ids
                if verification_ids is not None
                else self.committee_ids[: cfg.threshold]
            )
        )
        if len(self.verification_ids) != cfg.threshold:
            raise ValueError("verification identifier count must equal threshold")
        if len(set(self.verification_ids)) != len(self.verification_ids):
            raise ValueError("verification identifiers must be unique")
        if not set(self.verification_ids).issubset(self.committee_ids):
            raise ValueError("verification identifiers must belong to the committee")
        self.mask_powers = [cfg.beta**ell for ell in range(cfg.limbs)]
        self.x_points = list(range(1, cfg.committee_size + 1))
        self._committee_positions = {
            committee_id: position
            for position, committee_id in enumerate(self.committee_ids)
        }
        self.b_a = math.ceil(math.log2(2 * cfg.amax + 1))
        self.b_s = math.ceil(math.log2(2 * cfg.r0 + 1))

    def prove(self, statement: SnipStatement, witness: SnipWitness, seed: int) -> SnipProofResult:
        constraints = self.evaluate(statement, witness)
        relation_digest = self._relation_digest(
            statement=statement,
            output=constraints.output,
            residual=constraints.verification_residual,
        )
        committee_material = self.share_outputs(
            sigma=constraints.verification_residual,
            z=constraints.output,
            seed=seed,
        )
        return SnipProofResult(
            relation_digest=relation_digest,
            constraints=constraints,
            committee_material=committee_material,
        )

    def check(
        self,
        *,
        statement: SnipStatement,
        committee_id: int,
        challenge: int,
        material: SnipCommitteeMaterial,
    ) -> SnipCheckResult:
        """Perform the local challenge-dependent SNIP check for one verifier."""

        if tuple(statement.committee_ids) != self.verification_ids:
            raise ValueError("statement is bound to a different verification set")
        committee_id = int(committee_id)
        if committee_id not in self.verification_ids:
            raise ValueError("committee member is not selected for this round")
        challenge = int(challenge)
        if not 0 <= challenge < self.cfg.field_modulus:
            raise ValueError("challenge is outside the SNIP field")
        if len(material.z_shares) != self.output_count:
            raise ValueError("designated output share has wrong dimension")

        sigma_share = (
            challenge * int(material.sigma_share)
        ) % self.cfg.field_modulus
        return SnipCheckResult(
            sigma_share=sigma_share,
            z_shares=[int(value) for value in material.z_shares],
        )

    def evaluate(self, statement: SnipStatement, witness: SnipWitness) -> SnipConstraintResult:
        self._validate_shapes(statement, witness)

        residual_digest = hashlib.sha256()
        residual_digest.update(str(statement.sid).encode("utf-8"))
        residual_digest.update(int(statement.client_id).to_bytes(8, "little", signed=False))
        failed_constraints = 0
        linear_constraints = 0
        multiplication_constraints = 0

        def check_residual(value: int) -> None:
            nonlocal failed_constraints
            value = int(value)
            if value % self.cfg.field_modulus != 0:
                failed_constraints += 1
                residual_digest.update(str(value).encode("ascii"))
                residual_digest.update(b"|")

        mask = self.compose_mask(witness.limbs)
        for r in range(self.dimension):
            check_residual(
                witness.x[r]
                - statement.y[r]
                + witness.z[r]
                + witness.delta[r] * self.cfg.modulus
            )
            linear_constraints += 1

            check_residual(witness.delta[r] * (witness.delta[r] - 1))
            multiplication_constraints += 1

            range_residuals, range_linear, range_mult = self._range_residuals(
                witness.x[r] + self.cfg.amax,
                2 * self.cfg.amax,
            )
            for residual in range_residuals:
                check_residual(residual)
            linear_constraints += range_linear
            multiplication_constraints += range_mult

        for ell in range(self.cfg.limbs):
            for r in range(self.dimension):
                check_residual(witness.shifted[ell][r] - witness.limbs[ell][r] - self.cfg.r0)
                linear_constraints += 1

                range_residuals, range_linear, range_mult = self._range_residuals(
                    witness.shifted[ell][r],
                    2 * self.cfg.r0,
                )
                for residual in range_residuals:
                    check_residual(residual)
                linear_constraints += range_linear
                multiplication_constraints += range_mult

        for left, right in zip(witness.z, mask):
            check_residual(left - right)
            linear_constraints += 1

        verification_residual = 0
        if failed_constraints != 0:
            verification_residual = int.from_bytes(residual_digest.digest(), "little") % self.cfg.field_modulus
            verification_residual = verification_residual or 1
        return SnipConstraintResult(
            satisfied=(failed_constraints == 0),
            verification_residual=verification_residual,
            failed_constraints=failed_constraints,
            linear_constraints=linear_constraints,
            multiplication_constraints=multiplication_constraints,
            output=[int(value) for value in witness.z],
        )

    def share_outputs(self, *, sigma: int, z: Sequence[int], seed: int) -> list[SnipCommitteeMaterial]:
        secrets = [int(sigma) % self.cfg.field_modulus] + [int(value) % self.cfg.field_modulus for value in z]
        shares = self._shamir_share_vector(secrets, seed=seed)
        materials: list[SnipCommitteeMaterial] = []
        for h in range(self.cfg.committee_size):
            materials.append(SnipCommitteeMaterial(sigma_share=shares[h][0], z_shares=shares[h][1:]))
        return materials

    def reconstruct_scalar(
        self,
        shares: Sequence[int],
        share_ids: Sequence[int] | None = None,
    ) -> int:
        ids = tuple(int(value) for value in (share_ids or self.verification_ids))
        if len(shares) != len(ids):
            raise ValueError("share count does not match reconstruction set")
        coeffs = self._lagrange_coefficients(ids)
        total = 0
        for coeff, share in zip(coeffs, shares):
            total = (total + coeff * int(share)) % self.cfg.field_modulus
        return total

    def reconstruct_vector(
        self,
        shares: Sequence[Sequence[int]],
        share_ids: Sequence[int] | None = None,
    ) -> list[int]:
        ids = tuple(int(value) for value in (share_ids or self.verification_ids))
        if len(shares) != len(ids):
            raise ValueError("share count does not match reconstruction set")
        if not shares:
            raise ValueError("at least one share is required")
        coeffs = self._lagrange_coefficients(ids)
        total = [0 for _ in range(len(shares[0]))]
        for coeff, share_vec in zip(coeffs, shares):
            total = [
                (current + coeff * int(share)) % self.cfg.field_modulus
                for current, share in zip(total, share_vec)
            ]
        return total

    def compose_mask(self, limbs: Sequence[Sequence[int]]) -> list[int]:
        result = [0 for _ in range(self.dimension)]
        for ell, power in enumerate(self.mask_powers):
            result = [value + int(limb) * power for value, limb in zip(result, limbs[ell])]
        return result

    def _range_residuals(self, value: int, upper: int) -> tuple[list[int], int, int]:
        bit_length = math.ceil(math.log2(int(upper) + 1))
        modulus = 1 << bit_length
        value_int = int(value)
        left_bits = self._bits(value_int % modulus, bit_length)
        right_bits = self._bits((int(upper) - value_int) % modulus, bit_length)

        residuals: list[int] = []
        for bit in left_bits:
            residuals.append(bit * (bit - 1))
        for bit in right_bits:
            residuals.append(bit * (bit - 1))

        left_sum = sum((1 << j) * bit for j, bit in enumerate(left_bits))
        right_sum = sum((1 << j) * bit for j, bit in enumerate(right_bits))
        residuals.append(value_int - left_sum)
        residuals.append(int(upper) - value_int - right_sum)
        return residuals, 2, 2 * bit_length

    def _validate_shapes(self, statement: SnipStatement, witness: SnipWitness) -> None:
        if statement.theta_digest != self.theta_digest:
            raise ValueError("statement is bound to different protocol parameters")
        if tuple(statement.committee_ids) != self.verification_ids:
            raise ValueError("statement is bound to a different verification set")
        if len(statement.y) != self.dimension:
            raise ValueError("statement y has wrong dimension")
        if len(witness.x) != self.dimension:
            raise ValueError("witness x has wrong dimension")
        if len(witness.delta) != self.dimension:
            raise ValueError("witness delta has wrong dimension")
        if len(witness.limbs) != self.cfg.limbs:
            raise ValueError("witness limbs have wrong limb count")
        if len(witness.shifted) != self.cfg.limbs:
            raise ValueError("witness shifted values have wrong limb count")
        for row in witness.limbs:
            if len(row) != self.dimension:
                raise ValueError("witness limb row has wrong dimension")
        for row in witness.shifted:
            if len(row) != self.dimension:
                raise ValueError("witness shifted row has wrong dimension")
        if len(witness.z) != self.output_count:
            raise ValueError("witness output vector has wrong length")

    def _shamir_share_vector(self, secrets: Sequence[int], seed: int) -> list[list[int]]:
        rng = random.Random(int(seed))
        coeffs = [
            [rng.randrange(self.cfg.field_modulus) for _ in range(len(secrets))]
            for _ in range(self.cfg.threshold - 1)
        ]
        shares: list[list[int]] = []
        for x_point in self.x_points:
            row = [int(secret) % self.cfg.field_modulus for secret in secrets]
            power = 1
            for coeff in coeffs:
                power = (power * x_point) % self.cfg.field_modulus
                row = [(value + power * c) % self.cfg.field_modulus for value, c in zip(row, coeff)]
            shares.append(row)
        return shares

    def _lagrange_coefficients(self, share_ids: Sequence[int]) -> list[int]:
        ids = tuple(int(value) for value in share_ids)
        if len(ids) < self.cfg.threshold:
            raise ValueError("insufficient shares for threshold reconstruction")
        if len(set(ids)) != len(ids):
            raise ValueError("reconstruction identifiers must be unique")
        if not set(ids).issubset(self.committee_ids):
            raise ValueError("reconstruction identifier is outside the committee")
        points = [
            self.x_points[self._committee_positions[committee_id]]
            for committee_id in ids
        ]
        coeffs: list[int] = []
        q = self.cfg.field_modulus
        for i, xi in enumerate(points):
            num = 1
            den = 1
            for j, xj in enumerate(points):
                if i == j:
                    continue
                num = (num * (-xj)) % q
                den = (den * (xi - xj)) % q
            coeffs.append((num * pow(den % q, -1, q)) % q)
        return coeffs

    def _relation_digest(
        self,
        *,
        statement: SnipStatement,
        output: Sequence[int],
        residual: int,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"VMASK-relation-v1")
        digest.update(bytes.fromhex(statement.theta_digest))
        sid = str(statement.sid).encode("utf-8")
        digest.update(len(sid).to_bytes(4, "big", signed=False))
        digest.update(sid)
        digest.update(len(statement.committee_ids).to_bytes(4, "big", signed=False))
        for committee_id in statement.committee_ids:
            digest.update(int(committee_id).to_bytes(8, "big", signed=False))
        digest.update(int(statement.client_id).to_bytes(8, "big", signed=False))
        digest.update(len(statement.y).to_bytes(8, "big", signed=False))
        for value in statement.y:
            encoded = int(value) % self.cfg.modulus
            digest.update(encoded.to_bytes(16, "big", signed=False))
        digest.update(len(output).to_bytes(8, "big", signed=False))
        for value in output:
            encoded = int(value) % self.cfg.field_modulus
            digest.update(encoded.to_bytes(24, "big", signed=False))
        encoded_residual = int(residual) % self.cfg.field_modulus
        digest.update(encoded_residual.to_bytes(24, "big", signed=False))
        return digest.hexdigest()

    @staticmethod
    def _bits(value: int, bit_length: int) -> list[int]:
        return [(int(value) >> j) & 1 for j in range(bit_length)]
