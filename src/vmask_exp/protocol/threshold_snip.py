from __future__ import annotations

import hashlib
import random
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from .fast_poly import (
    ConsecutiveInterpolator,
    FlintPolynomialBackend,
    consecutive_lagrange_weights,
)

from . import prio_snip_reference as reference
from .config import ProtocolConfig


BACKEND_NAME = "complete_threshold_snip_prio_v1"


@dataclass(frozen=True)
class PublicInterpolationCacheInfo:
    field_modulus: int
    multiplication_count: int
    point_count: int
    setup_seconds: float


@dataclass
class _PublicInterpolationCacheEntry:
    info: PublicInterpolationCacheInfo
    interpolator: ConsecutiveInterpolator
    use_lock: object


_PUBLIC_INTERPOLATION_CACHE_LOCK = threading.RLock()
_PUBLIC_INTERPOLATION_CACHE: dict[
    tuple[int, int],
    _PublicInterpolationCacheEntry,
] = {}


def clear_public_interpolation_cache() -> None:

    with _PUBLIC_INTERPOLATION_CACHE_LOCK:
        _PUBLIC_INTERPOLATION_CACHE.clear()


def public_interpolation_cache_info() -> tuple[PublicInterpolationCacheInfo, ...]:

    with _PUBLIC_INTERPOLATION_CACHE_LOCK:
        return tuple(entry.info for entry in _PUBLIC_INTERPOLATION_CACHE.values())


def prepare_public_interpolation(
    field_modulus: int,
    multiplication_count: int,
) -> PublicInterpolationCacheInfo:

    return _public_interpolator_entry(
        field_modulus,
        multiplication_count,
    ).info


def _public_interpolator_entry(
    field_modulus: int,
    multiplication_count: int,
) -> _PublicInterpolationCacheEntry:
    q = int(field_modulus)
    m = int(multiplication_count)
    point_count = m + 1
    key = (q, point_count)
    with _PUBLIC_INTERPOLATION_CACHE_LOCK:
        cached = _PUBLIC_INTERPOLATION_CACHE.get(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        interpolator = ConsecutiveInterpolator(q, point_count)
        info = PublicInterpolationCacheInfo(
            field_modulus=q,
            multiplication_count=m,
            point_count=point_count,
            setup_seconds=time.perf_counter() - started,
        )
        entry = _PublicInterpolationCacheEntry(
            info=info,
            interpolator=interpolator,
            use_lock=threading.RLock(),
        )


        _PUBLIC_INTERPOLATION_CACHE.clear()
        _PUBLIC_INTERPOLATION_CACHE[key] = entry
        return entry


@dataclass(frozen=True)
class CompactShareEvaluation:
    multiplication_left: tuple[int, ...]
    multiplication_right: tuple[int, ...]
    constraints: tuple[int, ...]


@dataclass(frozen=True)
class VerifierTiming:
    committee_id: int
    phase_one_seconds: float
    phase_two_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.phase_one_seconds + self.phase_two_seconds


@dataclass(frozen=True)
class FullVerificationMetrics:
    result: reference.PrioVerificationResult
    total_seconds: float
    challenge_seconds: float
    public_weight_seconds: float
    d_e_opening_seconds: float
    residual_reconstruction_seconds: float
    designated_reconstruction_seconds: float
    verifier_timings: tuple[VerifierTiming, ...]
    serial_committee_work_seconds: float
    parallel_critical_path_estimate_seconds: float


def evaluate_share_with_gate_outputs(
    circuit: reference.ArithmeticCircuit,
    input_share: Sequence[int],
    multiplication_outputs: Sequence[int],
    server_count: int,
) -> CompactShareEvaluation:
    if len(input_share) != circuit.input_count:
        raise ValueError("circuit input-share length mismatch")
    if len(multiplication_outputs) != circuit.multiplication_count:
        raise ValueError("multiplication-output length mismatch")
    q = circuit.field_modulus
    inverse_servers = pow(int(server_count), -1, q)
    values: list[int] = []
    left_values: list[int] = []
    right_values: list[int] = []
    for node in circuit.nodes:
        if node.operation == "input":
            value = int(input_share[node.input_index]) % q
        elif node.operation == "constant":
            value = int(node.value) * inverse_servers % q
        elif node.operation == "add":
            value = (values[node.left] + values[node.right]) % q
        elif node.operation == "sub":
            value = (values[node.left] - values[node.right]) % q
        elif node.operation == "scale":
            value = values[node.left] * int(node.value) % q
        elif node.operation == "mul":
            left = values[node.left]
            right = values[node.right]
            value = int(multiplication_outputs[node.multiplication_index - 1]) % q
            left_values.append(left)
            right_values.append(right)
        else:
            raise ValueError("unknown circuit operation")
        values.append(value)
    return CompactShareEvaluation(
        multiplication_left=tuple(left_values),
        multiplication_right=tuple(right_values),
        constraints=tuple(values[index] for index in circuit.constraint_nodes),
    )


def prove_full(
    cfg: ProtocolConfig,
    statement: reference.PrioStatement,
    witness: reference.VMaskWitnessValues,
    seed: int | None = None,
) -> reference.PrioProofBundle:

    reference.validate_statement(cfg, statement)
    circuit, layout = reference.build_vmask_circuit(cfg, statement)
    expanded = reference.expand_witness(cfg, statement, witness)
    evaluation = circuit.evaluate(expanded)
    q = cfg.field_modulus
    rng = secrets.SystemRandom() if seed is None else random.Random(int(seed))
    f_zero = rng.randrange(q)
    g_zero = rng.randrange(q)

    interpolation_entry = _public_interpolator_entry(
        q,
        circuit.multiplication_count,
    )


    with interpolation_entry.use_lock:
        f_polynomial = interpolation_entry.interpolator.interpolate_poly(
            (f_zero,) + evaluation.multiplication_left
        )
        g_polynomial = interpolation_entry.interpolator.interpolate_poly(
            (g_zero,) + evaluation.multiplication_right
        )
    backend = FlintPolynomialBackend(q)
    expected_h = 2 * circuit.multiplication_count + 1
    h_coefficients = backend.multiply(
        f_polynomial,
        g_polynomial,
        expected_length=expected_h,
    )

    triple_a = rng.randrange(q)
    triple_b = rng.randrange(q)
    triple_c = triple_a * triple_b % q
    proof_values = (
        f_zero,
        g_zero,
        *h_coefficients,
        triple_a,
        triple_b,
        triple_c,
    )

    committee_positions = {
        identifier: index + 1
        for index, identifier in enumerate(statement.committee_ids)
    }
    selected_positions = tuple(
        committee_positions[identifier]
        for identifier in statement.verification_ids
    )
    designated = expanded[layout.z]
    designated_shares = reference.shamir_share_vector(
        designated,
        cfg.committee_size,
        cfg.threshold,
        q,
        rng,
    )
    non_designated = tuple(expanded[index] for index in layout.non_z_indices)
    non_designated_shares = reference.additive_share_vector(
        non_designated,
        cfg.threshold,
        q,
        rng,
    )
    proof_shares = reference.additive_share_vector(
        proof_values,
        cfg.threshold,
        q,
        rng,
    )
    selected_order = {
        identifier: index
        for index, identifier in enumerate(statement.verification_ids)
    }
    packets: list[reference.PrioPacket] = []
    for position, committee_id in enumerate(statement.committee_ids, start=1):
        if committee_id in selected_order:
            selected_index = selected_order[committee_id]
            proof_share = reference.parse_proof_share(
                proof_shares[selected_index],
                circuit.multiplication_count,
            )
            non_designated_share = non_designated_shares[selected_index]
        else:
            proof_share = None
            non_designated_share = None
        packets.append(
            reference.PrioPacket(
                committee_id=committee_id,
                share_x=position,
                designated_shamir_share=designated_shares[position - 1],
                non_designated_additive_share=non_designated_share,
                proof_share=proof_share,
                commitment_randomness=secrets.token_bytes(reference.SALT_BYTES),
            )
        )
    digest = reference.statement_digest(statement)
    commitments = tuple(
        reference.commit_packet(digest, packet, q)
        for packet in packets
    )
    adapter_coefficients = reference.lagrange_coefficients_zero(
        selected_positions,
        q,
    )
    if any(value == 0 for value in adapter_coefficients):
        raise AssertionError("selected-set adapter coefficient must be nonzero")
    return reference.PrioProofBundle(
        statement=statement,
        packets=tuple(packets),
        commitments=commitments,
        relation_satisfied=all(value % q == 0 for value in evaluation.constraints),
        multiplication_count=circuit.multiplication_count,
        constraint_count=circuit.constraint_count,
        witness_length=layout.input_count,
    )


def verify_full(
    cfg: ProtocolConfig,
    bundle: reference.PrioProofBundle,
    beacon: bytes,
) -> FullVerificationMetrics:
    total_start = time.perf_counter()
    statement = bundle.statement
    try:
        reference.validate_statement(cfg, statement)
        circuit, layout = reference.build_vmask_circuit(cfg, statement)
    except (ValueError, OverflowError):
        return _rejected_metrics(total_start, False)
    q = cfg.field_modulus
    if (
        len(bundle.packets) != cfg.committee_size
        or len(bundle.commitments) != cfg.committee_size
        or circuit.multiplication_count != bundle.multiplication_count
        or circuit.constraint_count != bundle.constraint_count
        or layout.input_count != bundle.witness_length
    ):
        return _rejected_metrics(total_start, False)
    digest = reference.statement_digest(statement)
    packet_by_id: dict[int, reference.PrioPacket] = {}
    commitment_by_id: dict[int, bytes] = {}
    for expected_position, (expected_id, packet, commitment) in enumerate(
        zip(statement.committee_ids, bundle.packets, bundle.commitments),
        start=1,
    ):
        if (
            packet.committee_id != expected_id
            or packet.share_x != expected_position
        ):
            return _rejected_metrics(total_start, False)
        packet_by_id[packet.committee_id] = packet
        commitment_by_id[packet.committee_id] = commitment
    selected = tuple(
        packet_by_id[identifier]
        for identifier in statement.verification_ids
    )
    if any(
        reference.commit_packet(digest, packet, q)
        != commitment_by_id[packet.committee_id]
        for packet in selected
    ):
        return _rejected_metrics(total_start, False)
    if any(
        packet.proof_share is None
        or packet.non_designated_additive_share is None
        for packet in selected
    ):
        return _rejected_metrics(total_start, True)
    if any(
        len(packet.proof_share.h_coefficients)
        != 2 * circuit.multiplication_count + 1
        for packet in selected
    ):
        return _rejected_metrics(total_start, True)
    share_points = tuple(packet.share_x for packet in selected)
    try:
        adapter_coefficients = reference.lagrange_coefficients_zero(
            share_points,
            q,
        )
    except ValueError:
        return _rejected_metrics(total_start, True)

    challenge_start = time.perf_counter()
    challenge_r, constraint_coefficients = derive_challenges_full(
        digest,
        bundle.commitments,
        beacon,
        q,
        circuit.multiplication_count,
        circuit.constraint_count,
    )
    challenge_seconds = time.perf_counter() - challenge_start

    polynomial_backend = FlintPolynomialBackend(q)
    gate_points = tuple(range(1, circuit.multiplication_count + 1))
    public_weight_start = time.perf_counter()
    try:
        interpolation_weights = consecutive_lagrange_weights(
            circuit.multiplication_count,
            challenge_r,
            q,
        )
    except (ValueError, ZeroDivisionError):
        return _rejected_metrics(total_start, True)
    public_weight_seconds = time.perf_counter() - public_weight_start
    d_shares: list[int] = []
    e_shares: list[int] = []
    phase_one_records = []
    verifier_phase_one: list[float] = []
    for packet, coefficient in zip(selected, adapter_coefficients):
        phase_start = time.perf_counter()
        try:
            input_share, _ = reference.adapted_input_share(
                packet,
                coefficient,
                layout,
                q,
            )
            proof_share = packet.proof_share
            multiplication_outputs = polynomial_backend.multipoint_evaluate(
                proof_share.h_coefficients,
                gate_points,
            )
            evaluation = evaluate_share_with_gate_outputs(
                circuit,
                input_share,
                multiplication_outputs,
                cfg.threshold,
            )
            f_r_share = reference.dot_product(
                interpolation_weights,
                (proof_share.f_zero,) + evaluation.multiplication_left,
                q,
            )
            g_r_share = reference.dot_product(
                interpolation_weights,
                (proof_share.g_zero,) + evaluation.multiplication_right,
                q,
            )
            h_r_share = polynomial_backend.evaluate(
                proof_share.h_coefficients,
                challenge_r,
            )
        except (ValueError, IndexError, TypeError, ZeroDivisionError):
            return _rejected_metrics(total_start, True)
        d_shares.append((f_r_share - proof_share.triple_a) % q)
        e_shares.append((challenge_r * g_r_share - proof_share.triple_b) % q)
        phase_one_records.append(
            (
                packet,
                proof_share,
                evaluation.constraints,
                h_r_share,
            )
        )
        verifier_phase_one.append(time.perf_counter() - phase_start)

    opening_start = time.perf_counter()
    opened_d = sum(d_shares) % q
    opened_e = sum(e_shares) % q
    d_e_opening_seconds = time.perf_counter() - opening_start

    inverse_servers = pow(cfg.threshold, -1, q)
    residual_reconstruction_shares: list[tuple[int, int]] = []
    verifier_phase_two: list[float] = []
    for (
        packet,
        proof_share,
        constraints,
        h_r_share,
    ), lagrange_coefficient in zip(phase_one_records, adapter_coefficients):
        phase_start = time.perf_counter()
        sigma = (
            opened_d * opened_e * inverse_servers
            + opened_d * proof_share.triple_b
            + opened_e * proof_share.triple_a
            + proof_share.triple_c
            - challenge_r * h_r_share
        ) % q
        relation_residual_share = reference.dot_product(
            constraint_coefficients,
            constraints,
            q,
        )
        inverse_lagrange = pow(int(lagrange_coefficient), -1, q)
        residual_reconstruction_shares.append(
            (
                sigma * inverse_lagrange % q,
                relation_residual_share * inverse_lagrange % q,
            )
        )
        verifier_phase_two.append(time.perf_counter() - phase_start)

    residual_start = time.perf_counter()
    polynomial_residual, constraint_residual = reference.reconstruct_shamir_vector(
        residual_reconstruction_shares,
        share_points,
        q,
    )
    residual_reconstruction_seconds = time.perf_counter() - residual_start

    accepted = (
        polynomial_residual == 0
        and constraint_residual == 0
    )
    result = reference.PrioVerificationResult(
        accepted=accepted,
        packet_openings_valid=True,
        polynomial_identity_residual=polynomial_residual,
        constraint_residual=constraint_residual,
        reconstructed_designated_output=None,
        adapter_output=None,
    )
    timings = tuple(
        VerifierTiming(
            committee_id=packet.committee_id,
            phase_one_seconds=phase_one,
            phase_two_seconds=phase_two,
        )
        for packet, phase_one, phase_two in zip(
            selected,
            verifier_phase_one,
            verifier_phase_two,
        )
    )
    serial_work = sum(timing.total_seconds for timing in timings)
    parallel_critical_path = (
        challenge_seconds
        + public_weight_seconds
        + max(verifier_phase_one, default=0.0)
        + d_e_opening_seconds
        + max(verifier_phase_two, default=0.0)
        + residual_reconstruction_seconds
    )
    return FullVerificationMetrics(
        result=result,
        total_seconds=time.perf_counter() - total_start,
        challenge_seconds=challenge_seconds,
        public_weight_seconds=public_weight_seconds,
        d_e_opening_seconds=d_e_opening_seconds,
        residual_reconstruction_seconds=residual_reconstruction_seconds,
        designated_reconstruction_seconds=0.0,
        verifier_timings=timings,
        serial_committee_work_seconds=serial_work,
        parallel_critical_path_estimate_seconds=parallel_critical_path,
    )


def _rejected_metrics(
    total_start: float,
    openings_valid: bool,
) -> FullVerificationMetrics:
    return FullVerificationMetrics(
        result=reference.rejected_result(openings_valid),
        total_seconds=time.perf_counter() - total_start,
        challenge_seconds=0.0,
        public_weight_seconds=0.0,
        d_e_opening_seconds=0.0,
        residual_reconstruction_seconds=0.0,
        designated_reconstruction_seconds=0.0,
        verifier_timings=(),
        serial_committee_work_seconds=0.0,
        parallel_critical_path_estimate_seconds=0.0,
    )


def deterministic_beacon(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


def derive_challenges_full(
    statement_hash: bytes,
    commitments: Sequence[bytes],
    beacon: bytes,
    field_modulus: int,
    multiplication_count: int,
    constraint_count: int,
) -> tuple[int, tuple[int, ...]]:

    if not beacon:
        raise ValueError("beacon must be nonempty")
    q = int(field_modulus)
    if q.bit_length() > 192:
        raise ValueError("Appendix-D challenge encoding supports at most 192 bits")
    material = (
        b"VMASK-PRIO-CHALLENGE-v1"
        + statement_hash
        + len(commitments).to_bytes(4, "big", signed=False)
        + b"".join(commitments)
        + len(beacon).to_bytes(4, "big", signed=False)
        + beacon
    )
    counter = 0

    def next_field_element() -> int:
        nonlocal counter
        while True:
            digest = hashlib.sha256(
                material + counter.to_bytes(8, "big", signed=False)
            ).digest()
            counter += 1
            candidate = int.from_bytes(digest[:24], "big", signed=False)
            if candidate < q:
                return candidate

    while True:
        evaluation_point = next_field_element()
        if not 1 <= evaluation_point <= int(multiplication_count):
            break
    coefficients = tuple(
        next_field_element() for _ in range(int(constraint_count))
    )
    return evaluation_point, coefficients
