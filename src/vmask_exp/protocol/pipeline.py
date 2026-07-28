from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .commitment import (
    derive_challenge,
    generate_challenge_seed,
    generate_packet_commitments,
    verify_packet_commitment,
)
from .config import ProtocolConfig, protocol_parameter_digest
from .snip import SnipCommitteeMaterial, SnipStatement, SnipWitness, VMaskSnip


@dataclass
class ProtectedSubmission:
    sid: str
    client_id: int
    y: list[int]


@dataclass
class CommitteeMaterial:
    sigma_share: int
    z_shares: list[int]
    salt: bytes


@dataclass
class VerificationEvidence:
    relation_digest: str
    statement_digest: bytes
    packet_commitments: tuple[bytes, ...]
    committee_material: list[CommitteeMaterial]


@dataclass
class ClientSubmission:
    gamma: ProtectedSubmission
    evidence: VerificationEvidence
    designated_output: list[int]
    accepted_by_relation: bool


@dataclass
class RoundResult:
    recovered_q_sum: object
    accepted_clients: list[int]
    rejected_clients: list[int]
    relation_failures: int
    packet_failures: int
    aggregate_equality: bool


class VMaskProtocol:

    def __init__(
        self,
        cfg: ProtocolConfig,
        dimension: int,
        max_clients: int,
        *,
        committee_ids: Sequence[int] | None = None,
        verification_ids: Sequence[int] | None = None,
    ):
        cfg.validate(dimension, max_clients)
        self.cfg = cfg
        self.dimension = int(dimension)
        self.max_clients = int(max_clients)
        self.committee_ids = tuple(
            int(value)
            for value in (
                committee_ids
                if committee_ids is not None
                else range(1, cfg.committee_size + 1)
            )
        )
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
        self._committee_positions = {
            committee_id: position
            for position, committee_id in enumerate(self.committee_ids)
        }
        self.theta_digest = protocol_parameter_digest(
            cfg,
            dimension=self.dimension,
            max_clients=self.max_clients,
        )
        self.output_count = self.dimension
        self.mask_powers = [cfg.beta**ell for ell in range(cfg.limbs)]
        self.snip = VMaskSnip(
            cfg,
            dimension,
            theta_digest=self.theta_digest,
            committee_ids=self.committee_ids,
            verification_ids=self.verification_ids,
        )

    def make_submission(
        self,
        *,
        sid: str,
        client_id: int,
        q_update,
        seed: int,
        malformed_direction: Sequence[int] | None = None,
    ) -> ClientSubmission:
        x = self._to_int_list(q_update)
        if len(x) != self.dimension:
            raise ValueError(f"expected dimension {self.dimension}, got {len(x)}")
        for value in x:
            if value < -self.cfg.amax or value > self.cfg.amax:
                raise ValueError("quantized update coordinate is outside [-Amax,Amax]")

        rng = random.Random(int(seed))
        limbs = [
            [rng.randint(-self.cfg.r0, self.cfg.r0) for _ in range(self.dimension)]
            for _ in range(self.cfg.limbs)
        ]
        if malformed_direction is not None:
            direction = [1 if int(v) >= 0 else -1 for v in malformed_direction]
            if len(direction) != self.dimension:
                raise ValueError("malformed direction has wrong dimension")
            for r, sign in enumerate(direction):
                limbs[0][r] = -(self.cfg.amax + 1) * sign

        mask = self._compose_mask(limbs)
        if malformed_direction is None:
            y_source = [x[r] + mask[r] for r in range(self.dimension)]
        else:
            # AMM construction: use the opposite orientation for the lowest
            # limb in the public masked update while keeping recovery values
            # derived from the submitted limbs.
            y_source = []
            for r in range(self.dimension):
                high = mask[r] - limbs[0][r]
                y_source.append(x[r] - limbs[0][r] + high)

        delta = [1 if value < 0 else 0 for value in y_source]
        y = [value + delta[r] * self.cfg.modulus for r, value in enumerate(y_source)]
        shifted = [[limbs[ell][r] + self.cfg.r0 for r in range(self.dimension)] for ell in range(self.cfg.limbs)]
        z = list(mask)
        statement = SnipStatement(
            theta_digest=self.theta_digest,
            sid=sid,
            committee_ids=self.verification_ids,
            client_id=client_id,
            y=y,
        )
        witness = SnipWitness(x=x, limbs=limbs, shifted=shifted, delta=delta, z=z)
        proof = self.snip.prove(statement, witness, seed=seed + 7919)
        gamma = ProtectedSubmission(sid=sid, client_id=client_id, y=y)
        packet_payloads = [
            (m.z_shares, m.sigma_share)
            for m in proof.committee_material
        ]
        commitment_bundle = generate_packet_commitments(
            statement,
            packet_payloads,
            self.cfg,
        )
        materials = [
            CommitteeMaterial(
                sigma_share=m.sigma_share,
                z_shares=m.z_shares,
                salt=commitment_bundle.salts[h],
            )
            for h, m in enumerate(proof.committee_material)
        ]
        evidence = VerificationEvidence(
            relation_digest=proof.relation_digest,
            statement_digest=commitment_bundle.statement_digest,
            packet_commitments=commitment_bundle.commitments,
            committee_material=materials,
        )
        return ClientSubmission(
            gamma=gamma,
            evidence=evidence,
            designated_output=z,
            accepted_by_relation=proof.constraints.satisfied,
        )

    def execute_round(
        self,
        *,
        sid: str,
        q_updates: Sequence,
        seeds: Sequence[int],
        malformed_clients: Iterable[int] = (),
        malformed_direction: Sequence[int] | None = None,
    ) -> RoundResult:
        malformed = set(int(i) for i in malformed_clients)
        if len(q_updates) != len(seeds):
            raise ValueError("q_updates and seeds must have the same length")

        y_aggregate = [0 for _ in range(self.dimension)]
        aggregate_z_shares = [
            [0 for _ in range(self.output_count)]
            for _ in range(self.cfg.threshold)
        ]
        accepted: list[int] = []
        rejected: list[int] = []
        relation_failures = 0
        packet_failures = 0

        for client_id, q_update in enumerate(q_updates):
            direction = malformed_direction if client_id in malformed else None
            submission = self.make_submission(
                sid=sid,
                client_id=client_id,
                q_update=q_update,
                seed=int(seeds[client_id]),
                malformed_direction=direction,
            )
            statement = SnipStatement(
                theta_digest=self.theta_digest,
                sid=submission.gamma.sid,
                committee_ids=self.verification_ids,
                client_id=submission.gamma.client_id,
                y=submission.gamma.y,
            )
            chi = generate_challenge_seed()
            rho = derive_challenge(
                statement=statement,
                commitments=submission.evidence.packet_commitments,
                chi=chi,
                cfg=self.cfg,
            )
            check_results = []
            packets_valid = True
            for committee_id in self.verification_ids:
                h = self._committee_positions[committee_id]
                material = submission.evidence.committee_material[h]
                packet_valid = verify_packet_commitment(
                    statement=statement,
                    committee_position=h + 1,
                    z_shares=material.z_shares,
                    verification_share=material.sigma_share,
                    salt=material.salt,
                    commitment=submission.evidence.packet_commitments[h],
                    cfg=self.cfg,
                )
                if not packet_valid:
                    packets_valid = False
                    break
                check_results.append(
                    self.snip.check(
                        statement=statement,
                        committee_id=committee_id,
                        challenge=rho,
                        material=SnipCommitteeMaterial(
                            sigma_share=material.sigma_share,
                            z_shares=material.z_shares,
                        ),
                    )
                )
            if not packets_valid:
                rejected.append(client_id)
                packet_failures += 1
                continue
            sigma = self.snip.reconstruct_scalar(
                [result.sigma_share for result in check_results],
                self.verification_ids,
            )
            if sigma != 0:
                rejected.append(client_id)
                relation_failures += 1
                continue
            accepted.append(client_id)
            y_aggregate = [
                (current + value) % self.cfg.modulus
                for current, value in zip(y_aggregate, submission.gamma.y)
            ]
            for h, result in enumerate(check_results):
                aggregate_z_shares[h] = [
                    (a + b) % self.cfg.field_modulus
                    for a, b in zip(aggregate_z_shares[h], result.z_shares)
                ]

        if not accepted:
            raise ValueError("no verified clients in round")

        z_aggregate = self.snip.reconstruct_vector(
            aggregate_z_shares,
            self.verification_ids,
        )
        aggregate_mask = [self._center_mod(value, self.cfg.field_modulus) for value in z_aggregate]
        recovered = [
            (y_aggregate[r] - aggregate_mask[r]) % self.cfg.modulus
            for r in range(self.dimension)
        ]
        centered = [self._center_mod(value, self.cfg.modulus) for value in recovered]

        reference = [0 for _ in range(self.dimension)]
        for client_id in accepted:
            q_list = self._to_int_list(q_updates[client_id])
            reference = [a + b for a, b in zip(reference, q_list)]
        aggregate_equality = centered == reference
        return RoundResult(
            recovered_q_sum=self._tensor_or_list(centered),
            accepted_clients=accepted,
            rejected_clients=rejected,
            relation_failures=relation_failures,
            packet_failures=packet_failures,
            aggregate_equality=aggregate_equality,
        )

    def recover_without_verification(
        self,
        *,
        q_updates: Sequence,
        malformed_clients: Iterable[int],
        malformed_direction: Sequence[int],
    ):
        malformed = set(int(i) for i in malformed_clients)
        direction = [1 if int(v) >= 0 else -1 for v in malformed_direction]
        aggregate = [0 for _ in range(self.dimension)]
        for q_update in q_updates:
            q_list = self._to_int_list(q_update)
            aggregate = [a + b for a, b in zip(aggregate, q_list)]
        residual_scale = 2 * (self.cfg.amax + 1) * len(malformed)
        aggregate = [value + residual_scale * direction[r] for r, value in enumerate(aggregate)]
        return self._tensor_or_list(aggregate)

    def _compose_mask(self, limbs: Sequence[Sequence[int]]) -> list[int]:
        result = [0 for _ in range(self.dimension)]
        for ell, power in enumerate(self.mask_powers):
            result = [value + int(limb) * power for value, limb in zip(result, limbs[ell])]
        return result

    @staticmethod
    def _to_int_list(values) -> list[int]:
        if hasattr(values, "detach"):
            return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]
        return [int(v) for v in values]

    @staticmethod
    def _center_mod(value: int, modulus: int) -> int:
        encoded = int(value) % int(modulus)
        return encoded - modulus if encoded >= modulus // 2 else encoded

    @staticmethod
    def _tensor_or_list(values: Sequence[int]):
        try:
            import torch
        except ModuleNotFoundError:
            return [int(v) for v in values]
        return torch.tensor([int(v) for v in values], dtype=torch.int64)
