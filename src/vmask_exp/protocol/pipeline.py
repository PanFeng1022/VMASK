from __future__ import annotations

import random
import secrets
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from .commitment import generate_challenge_seed
from .config import ProtocolConfig, protocol_parameter_digest
from .prio_snip_reference import (
    PrioPacket,
    PrioProofBundle,
    PrioStatement,
    VMaskWitnessValues,
    reconstruct_shamir_vector,
    statement_digest,
)
from .threshold_snip import (
    BACKEND_NAME,
    FullVerificationMetrics,
    prove_full,
    verify_full,
)

class CommitteeUnavailableError(ConnectionError):


    def __init__(self, committee_id: int):
        self.committee_id = int(committee_id)
        super().__init__(f"Committee member {self.committee_id} is unavailable")


@dataclass
class ProtectedSubmission:
    sid: str
    client_id: int
    y: list[int]


@dataclass
class VerificationEvidence:
    relation_digest: str
    statement_digest: bytes
    packet_commitments: tuple[bytes, ...]
    verifier_packets: tuple[PrioPacket, ...]
    multiplication_count: int | None = None
    constraint_count: int | None = None
    witness_length: int | None = None
    backend: str = BACKEND_NAME


@dataclass
class ClientSubmission:
    gamma: ProtectedSubmission
    evidence: VerificationEvidence
    witness_designated_output: list[int]
    relation_satisfied: bool
    snip_prove_seconds: float = 0.0


@dataclass(frozen=True)
class RoundTiming:
    client_prove_seconds: tuple[float, ...]
    client_check_seconds: tuple[float, ...]
    committee_serial_verification_seconds: float
    aggregate_recovery_seconds: float
    round_seconds: float


@dataclass
class RoundResult:
    recovered_q_sum: object | None
    accepted_clients: list[int]
    rejected_clients: list[int]
    relation_failures: int
    packet_failures: int
    aggregate_equality: bool | None
    released: bool = True
    aggregate_mask: list[int] | None = None
    release_reason: str | None = None
    timing: RoundTiming | None = None


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
        self.formal_snip_backend = BACKEND_NAME

    def make_submission(
        self,
        *,
        sid: str,
        client_id: int,
        q_update,
        seed: int | None = None,
        malformed_direction: Sequence[int] | None = None,
    ) -> ClientSubmission:
        x = self._to_int_list(q_update)
        if len(x) != self.dimension:
            raise ValueError(f"expected dimension {self.dimension}, got {len(x)}")
        for value in x:
            if value < -self.cfg.amax or value > self.cfg.amax:
                raise ValueError("quantized update coordinate is outside [-Amax,Amax]")


        rng = secrets.SystemRandom() if seed is None else random.Random(int(seed))
        limbs = [
            [rng.randint(-self.cfg.r0, self.cfg.r0) for _ in range(self.dimension)]
            for _ in range(self.cfg.limbs)
        ]
        if malformed_direction is not None:
            direction = [1 if int(value) >= 0 else -1 for value in malformed_direction]
            if len(direction) != self.dimension:
                raise ValueError("malformed direction has wrong dimension")
            for coordinate, sign in enumerate(direction):
                limbs[0][coordinate] = -(self.cfg.amax + 1) * sign

        mask = self._compose_mask(limbs)
        if malformed_direction is None:
            y_source = [x[index] + mask[index] for index in range(self.dimension)]
        else:
            y_source = []
            for coordinate in range(self.dimension):
                high = mask[coordinate] - limbs[0][coordinate]
                y_source.append(x[coordinate] - limbs[0][coordinate] + high)

        delta = [1 if value < 0 else 0 for value in y_source]
        y = [
            (value + delta[index] * self.cfg.modulus) % self.cfg.modulus
            for index, value in enumerate(y_source)
        ]
        shifted = [
            [limbs[limb][coordinate] + self.cfg.r0 for coordinate in range(self.dimension)]
            for limb in range(self.cfg.limbs)
        ]
        z = list(mask)
        statement = self._formal_statement(
            sid=sid,
            client_id=client_id,
            y=y,
        )
        witness = VMaskWitnessValues(
            x=tuple(x),
            limbs=tuple(tuple(row) for row in limbs),
            shifted=tuple(tuple(row) for row in shifted),
            delta=tuple(delta),
            z=tuple(z),
        )
        prove_start = time.perf_counter()
        proof = prove_full(
            self.cfg,
            statement,
            witness,
            seed=None if seed is None else int(seed) + 7919,
        )
        prove_seconds = time.perf_counter() - prove_start
        evidence = VerificationEvidence(
            relation_digest=self.theta_digest,
            statement_digest=statement_digest(statement),
            packet_commitments=proof.commitments,
            verifier_packets=proof.packets,
            multiplication_count=proof.multiplication_count,
            constraint_count=proof.constraint_count,
            witness_length=proof.witness_length,
            backend=BACKEND_NAME,
        )
        return ClientSubmission(
            gamma=ProtectedSubmission(sid=sid, client_id=client_id, y=y),
            evidence=evidence,
            witness_designated_output=z,
            relation_satisfied=proof.relation_satisfied,
            snip_prove_seconds=prove_seconds,
        )

    def execute_round(
        self,
        *,
        sid: str,
        q_updates: Sequence,
        seeds: Sequence[int | None] | None = None,
        malformed_clients: Iterable[int] = (),
        malformed_direction: Sequence[int] | None = None,
        available_committee_ids: Iterable[int] | None = None,
        received_submissions: Sequence[ClientSubmission | None] | None = None,
    ) -> RoundResult:

        round_start = time.perf_counter()
        malformed = set(int(index) for index in malformed_clients)
        seeds = (None,) * len(q_updates) if seeds is None else tuple(seeds)
        if len(q_updates) != len(seeds):
            raise ValueError("q_updates and seeds must have the same length")
        received = None if received_submissions is None else tuple(received_submissions)
        if received is not None:
            if len(received) != len(q_updates):
                raise ValueError("received_submissions must match the fixed client roster")
            if any(item is not None and not isinstance(item, ClientSubmission) for item in received):
                raise ValueError("received slots must contain ClientSubmission or None")

        available = set(
            self.committee_ids
            if available_committee_ids is None
            else (int(value) for value in available_committee_ids)
        )
        if not set(self.verification_ids).issubset(available):
            return self._abort_result(
                reason="verification_committee_unavailable",
                accepted=[],
                rejected=[],
                relation_failures=0,
                packet_failures=0,
                prove_times=(),
                check_times=(),
                committee_work=0.0,
                round_start=round_start,
            )

        y_aggregate = [0 for _ in range(self.dimension)]
        aggregate_designated_shares = {
            committee_id: [0 for _ in range(self.output_count)]
            for committee_id in self.verification_ids
        }
        accepted: list[int] = []
        rejected: list[int] = []
        relation_failures = 0
        packet_failures = 0
        prove_times: list[float] = []
        check_times: list[float] = []
        committee_work = 0.0

        for client_id, q_update in enumerate(q_updates):
            direction = malformed_direction if client_id in malformed else None
            submission = received[client_id] if received is not None else self.make_submission(
                sid=sid,
                client_id=client_id,
                q_update=q_update,
                seed=seeds[client_id],
                malformed_direction=direction,
            )
            if received is not None and submission is None:
                rejected.append(client_id)
                packet_failures += 1
                continue
            prove_times.append(submission.snip_prove_seconds)
            try:
                if (
                    submission.gamma.sid != sid
                    or submission.gamma.client_id != client_id
                    or submission.gamma.client_id in accepted
                ):
                    raise ValueError("submission does not match the expected round identity")
                bundle = self._formal_bundle(submission)
            except (TypeError, ValueError):
                rejected.append(client_id)
                packet_failures += 1
                continue



            check_start = time.perf_counter()
            beacon = generate_challenge_seed()
            try:
                verification: FullVerificationMetrics = verify_full(
                    self.cfg,
                    bundle,
                    beacon,
                )
            except CommitteeUnavailableError as error:
                if error.committee_id not in self.verification_ids:
                    raise


                return self._abort_result(
                    reason="verification_committee_unavailable",
                    accepted=accepted,
                    rejected=rejected,
                    relation_failures=relation_failures,
                    packet_failures=packet_failures,
                    prove_times=prove_times,
                    check_times=check_times,
                    committee_work=committee_work,
                    round_start=round_start,
                )
            check_times.append(time.perf_counter() - check_start)
            committee_work += verification.serial_committee_work_seconds
            if not verification.result.packet_openings_valid:
                rejected.append(client_id)
                packet_failures += 1
                continue
            if not verification.result.accepted:
                rejected.append(client_id)
                relation_failures += 1
                continue

            accepted.append(client_id)
            y_aggregate = [
                (current + value) % self.cfg.modulus
                for current, value in zip(y_aggregate, submission.gamma.y)
            ]
            packets_by_id = {
                packet.committee_id: packet for packet in bundle.packets
            }
            for committee_id in self.verification_ids:
                original_share = packets_by_id[committee_id].designated_shamir_share
                aggregate_designated_shares[committee_id] = [
                    (current + value) % self.cfg.field_modulus
                    for current, value in zip(
                        aggregate_designated_shares[committee_id],
                        original_share,
                    )
                ]

        if len(accepted) < self.cfg.kappa:
            return self._abort_result(
                reason="insufficient_verified_clients",
                accepted=accepted,
                rejected=rejected,
                relation_failures=relation_failures,
                packet_failures=packet_failures,
                prove_times=prove_times,
                check_times=check_times,
                committee_work=committee_work,
                round_start=round_start,
            )
        if not set(self.verification_ids).issubset(available):
            return self._abort_result(
                reason="verification_committee_unavailable",
                accepted=accepted,
                rejected=rejected,
                relation_failures=relation_failures,
                packet_failures=packet_failures,
                prove_times=prove_times,
                check_times=check_times,
                committee_work=committee_work,
                round_start=round_start,
            )

        recovery_start = time.perf_counter()
        share_points = tuple(
            self._committee_positions[committee_id] + 1
            for committee_id in self.verification_ids
        )
        try:
            z_aggregate = reconstruct_shamir_vector(
                tuple(
                    tuple(aggregate_designated_shares[committee_id])
                    for committee_id in self.verification_ids
                ),
                share_points,
                self.cfg.field_modulus,
            )
        except CommitteeUnavailableError as error:
            if error.committee_id not in self.verification_ids:
                raise


            return self._abort_result(
                reason="verification_committee_unavailable",
                accepted=accepted,
                rejected=rejected,
                relation_failures=relation_failures,
                packet_failures=packet_failures,
                prove_times=prove_times,
                check_times=check_times,
                committee_work=committee_work,
                round_start=round_start,
            )
        aggregate_mask = [
            self._center_mod(value, self.cfg.field_modulus)
            for value in z_aggregate
        ]
        recovered = [
            (y_aggregate[index] - aggregate_mask[index]) % self.cfg.modulus
            for index in range(self.dimension)
        ]
        centered = [self._center_mod(value, self.cfg.modulus) for value in recovered]
        aggregate_recovery_seconds = time.perf_counter() - recovery_start

        reference = [0 for _ in range(self.dimension)]
        for client_id in accepted:
            q_list = self._to_int_list(q_updates[client_id])
            reference = [a + b for a, b in zip(reference, q_list)]
        aggregate_equality = centered == reference
        timing = RoundTiming(
            client_prove_seconds=tuple(prove_times),
            client_check_seconds=tuple(check_times),
            committee_serial_verification_seconds=committee_work,
            aggregate_recovery_seconds=aggregate_recovery_seconds,
            round_seconds=time.perf_counter() - round_start,
        )
        return RoundResult(
            recovered_q_sum=self._tensor_or_list(centered),
            accepted_clients=accepted,
            rejected_clients=rejected,
            relation_failures=relation_failures,
            packet_failures=packet_failures,
            aggregate_equality=aggregate_equality,
            released=True,
            aggregate_mask=aggregate_mask,
            release_reason=None,
            timing=timing,
        )

    def recover_without_verification(
        self,
        *,
        q_updates: Sequence,
        malformed_clients: Iterable[int],
        malformed_direction: Sequence[int],
    ):
        malformed = set(int(index) for index in malformed_clients)
        direction = [1 if int(value) >= 0 else -1 for value in malformed_direction]
        aggregate = [0 for _ in range(self.dimension)]
        for q_update in q_updates:
            q_list = self._to_int_list(q_update)
            aggregate = [a + b for a, b in zip(aggregate, q_list)]
        residual_scale = 2 * (self.cfg.amax + 1) * len(malformed)
        aggregate = [
            value + residual_scale * direction[index]
            for index, value in enumerate(aggregate)
        ]
        return self._tensor_or_list(aggregate)

    def _formal_statement(
        self,
        *,
        sid: str,
        client_id: int,
        y: Sequence[int],
    ) -> PrioStatement:
        return PrioStatement(
            theta_digest=self.theta_digest,
            sid=str(sid),
            committee_ids=self.committee_ids,
            verification_ids=self.verification_ids,
            client_id=int(client_id),
            y=tuple(int(value) for value in y),
        )

    def _formal_bundle(self, submission: ClientSubmission) -> PrioProofBundle:
        evidence = submission.evidence
        if evidence.backend != BACKEND_NAME:
            raise ValueError("native execution requires the complete Threshold SNIP backend")
        if (
            evidence.multiplication_count is None
            or evidence.constraint_count is None
            or evidence.witness_length is None
        ):
            raise ValueError("complete Threshold SNIP metadata is missing")
        if not all(isinstance(packet, PrioPacket) for packet in evidence.verifier_packets):
            raise TypeError("complete Threshold SNIP packets are required")
        statement = self._formal_statement(
            sid=submission.gamma.sid,
            client_id=submission.gamma.client_id,
            y=submission.gamma.y,
        )
        digest = statement_digest(statement)
        if not secrets.compare_digest(digest, evidence.statement_digest):
            raise ValueError("statement digest mismatch")
        return PrioProofBundle(
            statement=statement,
            packets=tuple(evidence.verifier_packets),
            commitments=tuple(evidence.packet_commitments),
            relation_satisfied=False,
            multiplication_count=int(evidence.multiplication_count),
            constraint_count=int(evidence.constraint_count),
            witness_length=int(evidence.witness_length),
        )

    def _abort_result(
        self,
        *,
        reason: str,
        accepted: Sequence[int],
        rejected: Sequence[int],
        relation_failures: int,
        packet_failures: int,
        prove_times: Sequence[float],
        check_times: Sequence[float],
        committee_work: float,
        round_start: float,
    ) -> RoundResult:
        timing = RoundTiming(
            client_prove_seconds=tuple(float(value) for value in prove_times),
            client_check_seconds=tuple(float(value) for value in check_times),
            committee_serial_verification_seconds=float(committee_work),
            aggregate_recovery_seconds=0.0,
            round_seconds=time.perf_counter() - round_start,
        )
        return RoundResult(
            recovered_q_sum=None,
            accepted_clients=list(accepted),
            rejected_clients=list(rejected),
            relation_failures=int(relation_failures),
            packet_failures=int(packet_failures),
            aggregate_equality=None,
            released=False,
            aggregate_mask=None,
            release_reason=str(reason),
            timing=timing,
        )

    def _compose_mask(self, limbs: Sequence[Sequence[int]]) -> list[int]:
        result = [0 for _ in range(self.dimension)]
        for limb, power in zip(limbs, self.mask_powers):
            result = [
                value + int(coordinate) * power
                for value, coordinate in zip(result, limb)
            ]
        return result

    @staticmethod
    def _to_int_list(values) -> list[int]:
        if hasattr(values, "detach"):
            return [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
        return [int(value) for value in values]

    @staticmethod
    def _center_mod(value: int, modulus: int) -> int:
        encoded = int(value) % int(modulus)
        return encoded - modulus if encoded >= modulus // 2 else encoded

    @staticmethod
    def _tensor_or_list(values: Sequence[int]):
        try:
            import torch
        except ModuleNotFoundError:
            return [int(value) for value in values]
        return torch.tensor([int(value) for value in values], dtype=torch.int64)
