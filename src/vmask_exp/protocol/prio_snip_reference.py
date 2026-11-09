from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Sequence

from vmask_exp.protocol.config import ProtocolConfig


FIELD_ELEMENT_BYTES = 24
SALT_BYTES = 32


@dataclass(frozen=True)
class PrioStatement:
    theta_digest: str
    sid: str
    committee_ids: tuple[int, ...]
    verification_ids: tuple[int, ...]
    client_id: int
    y: tuple[int, ...]


@dataclass(frozen=True)
class VMaskWitnessValues:
    x: tuple[int, ...]
    limbs: tuple[tuple[int, ...], ...]
    shifted: tuple[tuple[int, ...], ...]
    delta: tuple[int, ...]
    z: tuple[int, ...]


@dataclass(frozen=True)
class WitnessLayout:
    dimension: int
    limbs: int
    b_a: int
    b_s: int
    x: slice
    mask_limbs: slice
    shifted: slice
    z: slice
    delta: slice
    x_bits: slice
    x_complement_bits: slice
    shifted_bits: slice
    shifted_complement_bits: slice
    input_count: int

    @property
    def z_indices(self) -> tuple[int, ...]:
        return tuple(range(self.z.start, self.z.stop))

    @property
    def non_z_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.input_count) if not self.z.start <= i < self.z.stop)


@dataclass(frozen=True)
class CircuitNode:
    operation: str
    left: int | None = None
    right: int | None = None
    value: int | None = None
    input_index: int | None = None
    multiplication_index: int | None = None


@dataclass(frozen=True)
class CircuitEvaluation:
    values: tuple[int, ...]
    multiplication_left: tuple[int, ...]
    multiplication_right: tuple[int, ...]
    multiplication_output: tuple[int, ...]
    constraints: tuple[int, ...]


class ArithmeticCircuit:
    def __init__(self, field_modulus: int, input_count: int):
        self.field_modulus = int(field_modulus)
        self.input_count = int(input_count)
        self.nodes: list[CircuitNode] = []
        self.constraint_nodes: list[int] = []
        self.multiplication_nodes: list[int] = []

    def input(self, index: int) -> int:
        if not 0 <= int(index) < self.input_count:
            raise ValueError("input index out of range")
        self.nodes.append(CircuitNode("input", input_index=int(index)))
        return len(self.nodes) - 1

    def constant(self, value: int) -> int:
        self.nodes.append(CircuitNode("constant", value=int(value) % self.field_modulus))
        return len(self.nodes) - 1

    def add(self, left: int, right: int) -> int:
        self.nodes.append(CircuitNode("add", left=int(left), right=int(right)))
        return len(self.nodes) - 1

    def sub(self, left: int, right: int) -> int:
        self.nodes.append(CircuitNode("sub", left=int(left), right=int(right)))
        return len(self.nodes) - 1

    def scale(self, node: int, scalar: int) -> int:
        self.nodes.append(CircuitNode("scale", left=int(node), value=int(scalar) % self.field_modulus))
        return len(self.nodes) - 1

    def mul(self, left: int, right: int) -> int:
        multiplication_index = len(self.multiplication_nodes) + 1
        self.nodes.append(
            CircuitNode(
                "mul",
                left=int(left),
                right=int(right),
                multiplication_index=multiplication_index,
            )
        )
        node_index = len(self.nodes) - 1
        self.multiplication_nodes.append(node_index)
        return node_index

    def constrain_zero(self, node: int) -> None:
        self.constraint_nodes.append(int(node))

    @property
    def multiplication_count(self) -> int:
        return len(self.multiplication_nodes)

    @property
    def constraint_count(self) -> int:
        return len(self.constraint_nodes)

    def evaluate(self, inputs: Sequence[int]) -> CircuitEvaluation:
        if len(inputs) != self.input_count:
            raise ValueError("circuit input length mismatch")
        q = self.field_modulus
        values: list[int] = []
        left_values: list[int] = []
        right_values: list[int] = []
        output_values: list[int] = []
        for node in self.nodes:
            if node.operation == "input":
                value = int(inputs[node.input_index]) % q
            elif node.operation == "constant":
                value = int(node.value) % q
            elif node.operation == "add":
                value = (values[node.left] + values[node.right]) % q
            elif node.operation == "sub":
                value = (values[node.left] - values[node.right]) % q
            elif node.operation == "scale":
                value = (values[node.left] * int(node.value)) % q
            elif node.operation == "mul":
                left = values[node.left]
                right = values[node.right]
                value = (left * right) % q
                left_values.append(left)
                right_values.append(right)
                output_values.append(value)
            else:
                raise ValueError("unknown circuit operation")
            values.append(value)
        return CircuitEvaluation(
            values=tuple(values),
            multiplication_left=tuple(left_values),
            multiplication_right=tuple(right_values),
            multiplication_output=tuple(output_values),
            constraints=tuple(values[index] for index in self.constraint_nodes),
        )

    def evaluate_share(
        self,
        input_share: Sequence[int],
        h_coefficient_share: Sequence[int],
        server_count: int,
    ) -> CircuitEvaluation:
        if len(input_share) != self.input_count:
            raise ValueError("circuit input-share length mismatch")
        expected_h = 2 * self.multiplication_count + 1
        if len(h_coefficient_share) != expected_h:
            raise ValueError("h coefficient-share length mismatch")
        q = self.field_modulus
        inv_servers = pow(int(server_count), -1, q)
        values: list[int] = []
        left_values: list[int] = []
        right_values: list[int] = []
        output_values: list[int] = []
        for node in self.nodes:
            if node.operation == "input":
                value = int(input_share[node.input_index]) % q
            elif node.operation == "constant":
                value = (int(node.value) * inv_servers) % q
            elif node.operation == "add":
                value = (values[node.left] + values[node.right]) % q
            elif node.operation == "sub":
                value = (values[node.left] - values[node.right]) % q
            elif node.operation == "scale":
                value = (values[node.left] * int(node.value)) % q
            elif node.operation == "mul":
                left = values[node.left]
                right = values[node.right]
                value = polynomial_evaluate(h_coefficient_share, node.multiplication_index, q)
                left_values.append(left)
                right_values.append(right)
                output_values.append(value)
            else:
                raise ValueError("unknown circuit operation")
            values.append(value)
        return CircuitEvaluation(
            values=tuple(values),
            multiplication_left=tuple(left_values),
            multiplication_right=tuple(right_values),
            multiplication_output=tuple(output_values),
            constraints=tuple(values[index] for index in self.constraint_nodes),
        )


@dataclass(frozen=True)
class PrioProofShare:
    f_zero: int
    g_zero: int
    h_coefficients: tuple[int, ...]
    triple_a: int
    triple_b: int
    triple_c: int


@dataclass(frozen=True)
class PrioPacket:
    committee_id: int
    share_x: int
    designated_shamir_share: tuple[int, ...]
    non_designated_additive_share: tuple[int, ...] | None
    proof_share: PrioProofShare | None
    commitment_randomness: bytes


@dataclass(frozen=True)
class PrioProofBundle:
    statement: PrioStatement
    packets: tuple[PrioPacket, ...]
    commitments: tuple[bytes, ...]
    relation_satisfied: bool
    multiplication_count: int
    constraint_count: int
    witness_length: int


@dataclass(frozen=True)
class PrioVerificationResult:
    accepted: bool
    packet_openings_valid: bool
    polynomial_identity_residual: int | None
    constraint_residual: int | None
    reconstructed_designated_output: tuple[int, ...] | None
    adapter_output: tuple[int, ...] | None


@dataclass(frozen=True)
class PrioCostEstimate:
    witness_field_elements: int
    multiplication_gates: int
    constraints: int
    proof_share_field_elements: int
    selected_packet_field_elements: int
    nonselected_packet_field_elements: int
    total_client_field_elements: int
    proof_bytes_per_selected_verifier: int
    selected_packet_bytes_without_framing: int
    total_client_bytes_without_framing: int
    logical_verifier_broadcast_field_elements: int
    point_to_point_verifier_field_elements: int


def witness_layout(cfg: ProtocolConfig, dimension: int) -> WitnessLayout:
    d = int(dimension)
    b_a = math.ceil(math.log2(2 * cfg.amax + 1))
    b_s = math.ceil(math.log2(2 * cfg.r0 + 1))
    offset = 0

    def take(length: int) -> slice:
        nonlocal offset
        result = slice(offset, offset + int(length))
        offset += int(length)
        return result

    return WitnessLayout(
        dimension=d,
        limbs=cfg.limbs,
        b_a=b_a,
        b_s=b_s,
        x=take(d),
        mask_limbs=take(cfg.limbs * d),
        shifted=take(cfg.limbs * d),
        z=take(d),
        delta=take(d),
        x_bits=take(d * b_a),
        x_complement_bits=take(d * b_a),
        shifted_bits=take(cfg.limbs * d * b_s),
        shifted_complement_bits=take(cfg.limbs * d * b_s),
        input_count=offset,
    )


def expand_witness(
    cfg: ProtocolConfig,
    statement: PrioStatement,
    witness: VMaskWitnessValues,
) -> tuple[int, ...]:
    d = len(statement.y)
    layout = witness_layout(cfg, d)
    if len(witness.x) != d or len(witness.z) != d or len(witness.delta) != d:
        raise ValueError("witness vector length mismatch")
    if len(witness.limbs) != cfg.limbs or len(witness.shifted) != cfg.limbs:
        raise ValueError("witness limb count mismatch")
    if any(len(row) != d for row in witness.limbs + witness.shifted):
        raise ValueError("witness limb dimension mismatch")
    flat: list[int] = []
    flat.extend(witness.x)
    flat.extend(value for row in witness.limbs for value in row)
    flat.extend(value for row in witness.shifted for value in row)
    flat.extend(witness.z)
    flat.extend(witness.delta)
    for value in witness.x:
        flat.extend(integer_bits(value + cfg.amax, layout.b_a))
    for value in witness.x:
        flat.extend(integer_bits(2 * cfg.amax - (value + cfg.amax), layout.b_a))
    for row in witness.shifted:
        for value in row:
            flat.extend(integer_bits(value, layout.b_s))
    for row in witness.shifted:
        for value in row:
            flat.extend(integer_bits(2 * cfg.r0 - value, layout.b_s))
    if len(flat) != layout.input_count:
        raise AssertionError("expanded witness length mismatch")
    return tuple(int(value) % cfg.field_modulus for value in flat)


def build_vmask_circuit(
    cfg: ProtocolConfig,
    statement: PrioStatement,
) -> tuple[ArithmeticCircuit, WitnessLayout]:
    d = len(statement.y)
    layout = witness_layout(cfg, d)
    circuit = ArithmeticCircuit(cfg.field_modulus, layout.input_count)
    inputs = [circuit.input(index) for index in range(layout.input_count)]

    def index_2d(region: slice, first: int, second: int, width: int) -> int:
        return region.start + first * width + second

    def index_3d(region: slice, first: int, second: int, third: int, width: int, depth: int) -> int:
        return region.start + (first * width + second) * depth + third

    one = circuit.constant(1)

    def add_many(nodes: Sequence[int]) -> int:
        if not nodes:
            return circuit.constant(0)
        result = int(nodes[0])
        for node in nodes[1:]:
            result = circuit.add(result, int(node))
        return result

    def constrain_range(value_node: int, upper: int, low_bits: Sequence[int], high_bits: Sequence[int]) -> None:
        for bit_node in tuple(low_bits) + tuple(high_bits):
            bit_minus_one = circuit.sub(bit_node, one)
            circuit.constrain_zero(circuit.mul(bit_node, bit_minus_one))
        low_sum = add_many([circuit.scale(node, 1 << bit) for bit, node in enumerate(low_bits)])
        high_sum = add_many([circuit.scale(node, 1 << bit) for bit, node in enumerate(high_bits)])
        circuit.constrain_zero(circuit.sub(value_node, low_sum))
        complement = circuit.sub(circuit.constant(upper), value_node)
        circuit.constrain_zero(circuit.sub(complement, high_sum))

    for coordinate in range(d):
        x_node = inputs[layout.x.start + coordinate]
        z_node = inputs[layout.z.start + coordinate]
        delta_node = inputs[layout.delta.start + coordinate]
        c1 = circuit.add(x_node, z_node)
        c1 = circuit.add(c1, circuit.scale(delta_node, cfg.modulus))
        c1 = circuit.sub(c1, circuit.constant(statement.y[coordinate]))
        circuit.constrain_zero(c1)
        circuit.constrain_zero(circuit.mul(delta_node, circuit.sub(delta_node, one)))
        x_shifted = circuit.add(x_node, circuit.constant(cfg.amax))
        x_bits = [inputs[layout.x_bits.start + coordinate * layout.b_a + bit] for bit in range(layout.b_a)]
        x_complement_bits = [
            inputs[layout.x_complement_bits.start + coordinate * layout.b_a + bit]
            for bit in range(layout.b_a)
        ]
        constrain_range(x_shifted, 2 * cfg.amax, x_bits, x_complement_bits)

        mask_terms: list[int] = []
        for limb in range(cfg.limbs):
            m_node = inputs[index_2d(layout.mask_limbs, limb, coordinate, d)]
            s_node = inputs[index_2d(layout.shifted, limb, coordinate, d)]
            c4 = circuit.sub(circuit.sub(s_node, m_node), circuit.constant(cfg.r0))
            circuit.constrain_zero(c4)
            s_bits = [
                inputs[index_3d(layout.shifted_bits, limb, coordinate, bit, d, layout.b_s)]
                for bit in range(layout.b_s)
            ]
            s_complement_bits = [
                inputs[
                    index_3d(
                        layout.shifted_complement_bits,
                        limb,
                        coordinate,
                        bit,
                        d,
                        layout.b_s,
                    )
                ]
                for bit in range(layout.b_s)
            ]
            constrain_range(s_node, 2 * cfg.r0, s_bits, s_complement_bits)
            mask_terms.append(circuit.scale(m_node, cfg.beta**limb))
        circuit.constrain_zero(circuit.sub(z_node, add_many(mask_terms)))

    return circuit, layout


def prove(
    cfg: ProtocolConfig,
    statement: PrioStatement,
    witness: VMaskWitnessValues,
    seed: int,
) -> PrioProofBundle:
    validate_statement(cfg, statement)
    circuit, layout = build_vmask_circuit(cfg, statement)
    expanded = expand_witness(cfg, statement, witness)
    evaluation = circuit.evaluate(expanded)
    q = cfg.field_modulus
    rng = random.Random(int(seed))
    f_zero = rng.randrange(q)
    g_zero = rng.randrange(q)
    points = tuple(range(circuit.multiplication_count + 1))
    f_coefficients = polynomial_interpolate(
        points,
        (f_zero,) + evaluation.multiplication_left,
        q,
    )
    g_coefficients = polynomial_interpolate(
        points,
        (g_zero,) + evaluation.multiplication_right,
        q,
    )
    h_coefficients = polynomial_multiply(f_coefficients, g_coefficients, q)
    expected_h = 2 * circuit.multiplication_count + 1
    h_coefficients = tuple(h_coefficients) + (0,) * (expected_h - len(h_coefficients))
    triple_a = rng.randrange(q)
    triple_b = rng.randrange(q)
    triple_c = (triple_a * triple_b) % q
    proof_values = (
        f_zero,
        g_zero,
        *h_coefficients,
        triple_a,
        triple_b,
        triple_c,
    )

    committee_positions = {identifier: index + 1 for index, identifier in enumerate(statement.committee_ids)}
    selected_positions = tuple(committee_positions[identifier] for identifier in statement.verification_ids)
    designated = expanded[layout.z]
    designated_shares = shamir_share_vector(
        designated,
        cfg.committee_size,
        cfg.threshold,
        q,
        rng,
    )
    non_designated = tuple(expanded[index] for index in layout.non_z_indices)
    non_designated_shares = additive_share_vector(non_designated, cfg.threshold, q, rng)
    proof_shares = additive_share_vector(proof_values, cfg.threshold, q, rng)
    selected_order = {identifier: index for index, identifier in enumerate(statement.verification_ids)}
    packets: list[PrioPacket] = []
    for position, committee_id in enumerate(statement.committee_ids, start=1):
        if committee_id in selected_order:
            selected_index = selected_order[committee_id]
            proof_share = parse_proof_share(proof_shares[selected_index], circuit.multiplication_count)
            non_designated_share = non_designated_shares[selected_index]
        else:
            proof_share = None
            non_designated_share = None
        packets.append(
            PrioPacket(
                committee_id=committee_id,
                share_x=position,
                designated_shamir_share=designated_shares[position - 1],
                non_designated_additive_share=non_designated_share,
                proof_share=proof_share,
                commitment_randomness=random_bytes(rng, SALT_BYTES),
            )
        )
    digest = statement_digest(statement)
    commitments = tuple(commit_packet(digest, packet, q) for packet in packets)
    adapter_coefficients = lagrange_coefficients_zero(selected_positions, q)
    if any(value == 0 for value in adapter_coefficients):
        raise AssertionError("selected-set adapter coefficient must be nonzero")
    return PrioProofBundle(
        statement=statement,
        packets=tuple(packets),
        commitments=commitments,
        relation_satisfied=all(value % q == 0 for value in evaluation.constraints),
        multiplication_count=circuit.multiplication_count,
        constraint_count=circuit.constraint_count,
        witness_length=layout.input_count,
    )


def verify(
    cfg: ProtocolConfig,
    bundle: PrioProofBundle,
    beacon: bytes,
) -> PrioVerificationResult:
    statement = bundle.statement
    try:
        validate_statement(cfg, statement)
        circuit, layout = build_vmask_circuit(cfg, statement)
    except (ValueError, OverflowError):
        return rejected_result(False)
    q = cfg.field_modulus
    if (
        len(bundle.packets) != cfg.committee_size
        or len(bundle.commitments) != cfg.committee_size
        or circuit.multiplication_count != bundle.multiplication_count
        or circuit.constraint_count != bundle.constraint_count
        or layout.input_count != bundle.witness_length
    ):
        return rejected_result(False)
    digest = statement_digest(statement)
    openings_valid = True
    packet_by_id: dict[int, PrioPacket] = {}
    for expected_id, packet, commitment in zip(statement.committee_ids, bundle.packets, bundle.commitments):
        if packet.committee_id != expected_id:
            openings_valid = False
            break
        if commit_packet(digest, packet, q) != commitment:
            openings_valid = False
            break
        packet_by_id[packet.committee_id] = packet
    if not openings_valid:
        return rejected_result(False)
    selected = tuple(packet_by_id[identifier] for identifier in statement.verification_ids)
    if any(packet.proof_share is None or packet.non_designated_additive_share is None for packet in selected):
        return rejected_result(True)
    share_points = tuple(packet.share_x for packet in selected)
    try:
        adapter_coefficients = lagrange_coefficients_zero(share_points, q)
    except ValueError:
        return rejected_result(True)
    challenge_r, constraint_coefficients = derive_challenges(
        digest,
        bundle.commitments,
        beacon,
        q,
        circuit.constraint_count,
    )
    interpolation_weights = lagrange_evaluation_weights(
        tuple(range(circuit.multiplication_count + 1)),
        challenge_r,
        q,
    )
    d_shares: list[int] = []
    e_shares: list[int] = []
    local_records: list[tuple[PrioPacket, CircuitEvaluation, int]] = []
    adapter_outputs: list[tuple[int, ...]] = []
    for packet, coefficient in zip(selected, adapter_coefficients):
        try:
            input_share, adapter_output = adapted_input_share(packet, coefficient, layout, q)
            proof_share = packet.proof_share
            evaluation = circuit.evaluate_share(
                input_share,
                proof_share.h_coefficients,
                cfg.threshold,
            )
        except (ValueError, IndexError, TypeError):
            return rejected_result(True)
        f_r_share = dot_product(
            interpolation_weights,
            (proof_share.f_zero,) + evaluation.multiplication_left,
            q,
        )
        g_r_share = dot_product(
            interpolation_weights,
            (proof_share.g_zero,) + evaluation.multiplication_right,
            q,
        )
        h_r_share = polynomial_evaluate(proof_share.h_coefficients, challenge_r, q)
        d_shares.append((f_r_share - proof_share.triple_a) % q)
        e_shares.append((challenge_r * g_r_share - proof_share.triple_b) % q)
        local_records.append((packet, evaluation, h_r_share))
        adapter_outputs.append(adapter_output)
    opened_d = sum(d_shares) % q
    opened_e = sum(e_shares) % q
    inv_servers = pow(cfg.threshold, -1, q)
    sigma_shares: list[int] = []
    constraint_shares: list[int] = []
    for packet, evaluation, h_r_share in local_records:
        proof_share = packet.proof_share
        sigma = (
            opened_d * opened_e * inv_servers
            + opened_d * proof_share.triple_b
            + opened_e * proof_share.triple_a
            + proof_share.triple_c
            - challenge_r * h_r_share
        ) % q
        sigma_shares.append(sigma)
        constraint_shares.append(
            dot_product(constraint_coefficients, evaluation.constraints, q)
        )
    polynomial_residual = sum(sigma_shares) % q
    constraint_residual = sum(constraint_shares) % q
    recovered_designated = reconstruct_shamir_vector(
        tuple(packet.designated_shamir_share for packet in selected),
        share_points,
        q,
    )
    adapter_output = tuple(
        sum(shares[coordinate] for shares in adapter_outputs) % q
        for coordinate in range(layout.dimension)
    )
    adapter_consistent = recovered_designated == adapter_output
    accepted = polynomial_residual == 0 and constraint_residual == 0 and adapter_consistent
    return PrioVerificationResult(
        accepted=accepted,
        packet_openings_valid=True,
        polynomial_identity_residual=polynomial_residual,
        constraint_residual=constraint_residual,
        reconstructed_designated_output=recovered_designated,
        adapter_output=adapter_output,
    )


def cost_estimate(
    cfg: ProtocolConfig,
    dimension: int,
) -> PrioCostEstimate:
    d = int(dimension)
    layout = witness_layout(cfg, d)
    b_a = layout.b_a
    b_s = layout.b_s
    multiplication_gates = d * (1 + 2 * b_a + 2 * cfg.limbs * b_s)
    constraints = d * (5 + 2 * b_a + cfg.limbs * (2 * b_s + 3))
    proof_share = 2 * multiplication_gates + 6
    selected_packet = layout.input_count + proof_share
    nonselected_packet = d
    total = (
        cfg.committee_size * d
        + cfg.threshold * ((layout.input_count - d) + proof_share)
    )
    logical_broadcast = 4 * cfg.threshold
    point_to_point = logical_broadcast * (cfg.threshold - 1)
    return PrioCostEstimate(
        witness_field_elements=layout.input_count,
        multiplication_gates=multiplication_gates,
        constraints=constraints,
        proof_share_field_elements=proof_share,
        selected_packet_field_elements=selected_packet,
        nonselected_packet_field_elements=nonselected_packet,
        total_client_field_elements=total,
        proof_bytes_per_selected_verifier=proof_share * FIELD_ELEMENT_BYTES,
        selected_packet_bytes_without_framing=selected_packet * FIELD_ELEMENT_BYTES,
        total_client_bytes_without_framing=total * FIELD_ELEMENT_BYTES,
        logical_verifier_broadcast_field_elements=logical_broadcast,
        point_to_point_verifier_field_elements=point_to_point,
    )


def replace_packet(
    cfg: ProtocolConfig,
    bundle: PrioProofBundle,
    packet_index: int,
    packet: PrioPacket,
    recommit: bool,
) -> PrioProofBundle:
    packets = list(bundle.packets)
    commitments = list(bundle.commitments)
    packets[int(packet_index)] = packet
    if recommit:
        commitments[int(packet_index)] = commit_packet(
            statement_digest(bundle.statement),
            packet,
            cfg.field_modulus,
        )
    return replace(bundle, packets=tuple(packets), commitments=tuple(commitments))


def validate_statement(cfg: ProtocolConfig, statement: PrioStatement) -> None:
    if len(statement.committee_ids) != cfg.committee_size:
        raise ValueError("committee identifier count mismatch")
    if len(statement.verification_ids) != cfg.threshold:
        raise ValueError("verification identifier count mismatch")
    if len(set(statement.committee_ids)) != len(statement.committee_ids):
        raise ValueError("committee identifiers must be unique")
    if len(set(statement.verification_ids)) != len(statement.verification_ids):
        raise ValueError("verification identifiers must be unique")
    if not set(statement.verification_ids).issubset(statement.committee_ids):
        raise ValueError("verification set is outside committee")
    if not statement.y:
        raise ValueError("statement dimension must be positive")
    if any(not 0 <= int(value) < cfg.modulus for value in statement.y):
        raise ValueError("public masked update is outside the aggregation ring")


def adapted_input_share(
    packet: PrioPacket,
    coefficient: int,
    layout: WitnessLayout,
    field_modulus: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if packet.non_designated_additive_share is None:
        raise ValueError("selected verifier lacks additive witness share")
    if len(packet.designated_shamir_share) != layout.dimension:
        raise ValueError("designated Shamir share dimension mismatch")
    if len(packet.non_designated_additive_share) != len(layout.non_z_indices):
        raise ValueError("non-designated additive share length mismatch")
    q = int(field_modulus)
    designated_additive = tuple(
        int(coefficient) * int(value) % q
        for value in packet.designated_shamir_share
    )
    non_designated = iter(packet.non_designated_additive_share)
    designated = iter(designated_additive)
    result: list[int] = []
    for index in range(layout.input_count):
        if layout.z.start <= index < layout.z.stop:
            result.append(next(designated))
        else:
            result.append(int(next(non_designated)) % q)
    return tuple(result), designated_additive


def statement_digest(statement: PrioStatement) -> bytes:
    payload = {
        "theta_digest": statement.theta_digest,
        "sid": statement.sid,
        "committee_ids": list(statement.committee_ids),
        "verification_ids": list(statement.verification_ids),
        "client_id": statement.client_id,
        "y": [str(value) for value in statement.y],
    }
    return hashlib.sha256(
        b"VMASK-PRIO-STMT-v1"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def commit_packet(statement_hash: bytes, packet: PrioPacket, field_modulus: int) -> bytes:
    payload = serialize_packet_payload(packet, field_modulus)
    return hashlib.sha256(
        b"VMASK-PRIO-PACKET-v1"
        + statement_hash
        + int(packet.committee_id).to_bytes(8, "big", signed=False)
        + packet.commitment_randomness
        + payload
    ).digest()


def serialize_packet_payload(packet: PrioPacket, field_modulus: int) -> bytes:
    payload = bytearray()
    payload.extend(int(packet.share_x).to_bytes(4, "big", signed=False))
    payload.extend(encode_vector(packet.designated_shamir_share, field_modulus))
    if packet.non_designated_additive_share is None or packet.proof_share is None:
        payload.extend(b"\x00")
        return bytes(payload)
    payload.extend(b"\x01")
    payload.extend(encode_vector(packet.non_designated_additive_share, field_modulus))
    proof = packet.proof_share
    payload.extend(encode_vector((proof.f_zero, proof.g_zero), field_modulus))
    payload.extend(encode_vector(proof.h_coefficients, field_modulus))
    payload.extend(
        encode_vector((proof.triple_a, proof.triple_b, proof.triple_c), field_modulus)
    )
    return bytes(payload)


def encode_vector(values: Sequence[int], field_modulus: int) -> bytes:
    payload = bytearray(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        encoded = int(value) % int(field_modulus)
        payload.extend(encoded.to_bytes(FIELD_ELEMENT_BYTES, "big", signed=False))
    return bytes(payload)


def derive_challenges(
    statement_hash: bytes,
    commitments: Sequence[bytes],
    beacon: bytes,
    field_modulus: int,
    constraint_count: int,
) -> tuple[int, tuple[int, ...]]:
    if not beacon:
        raise ValueError("beacon must be nonempty")
    seed = hashlib.sha256(
        b"VMASK-PRIO-CHALLENGE-v1"
        + statement_hash
        + len(commitments).to_bytes(4, "big", signed=False)
        + b"".join(commitments)
        + len(beacon).to_bytes(4, "big", signed=False)
        + beacon
    ).digest()
    evaluation_point = hash_to_field(seed, b"multiplication", 0, field_modulus)
    coefficients = tuple(
        hash_to_field(seed, b"constraint", index, field_modulus)
        for index in range(int(constraint_count))
    )
    return evaluation_point, coefficients


def hash_to_field(seed: bytes, label: bytes, index: int, field_modulus: int) -> int:
    q = int(field_modulus)
    limit = (1 << 256) - ((1 << 256) % q)
    counter = 0
    while True:
        digest = hashlib.sha256(
            b"VMASK-PRIO-H2F-v1"
            + seed
            + len(label).to_bytes(2, "big", signed=False)
            + label
            + int(index).to_bytes(8, "big", signed=False)
            + counter.to_bytes(4, "big", signed=False)
        ).digest()
        candidate = int.from_bytes(digest, "big", signed=False)
        if candidate < limit:
            return candidate % q
        counter += 1


def additive_share_vector(
    secrets: Sequence[int],
    share_count: int,
    field_modulus: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    q = int(field_modulus)
    rows = [[0 for _ in secrets] for _ in range(int(share_count))]
    for coordinate, secret in enumerate(secrets):
        total = 0
        for share_index in range(int(share_count) - 1):
            value = rng.randrange(q)
            rows[share_index][coordinate] = value
            total = (total + value) % q
        rows[-1][coordinate] = (int(secret) - total) % q
    return tuple(tuple(row) for row in rows)


def shamir_share_vector(
    secrets: Sequence[int],
    committee_size: int,
    threshold: int,
    field_modulus: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    q = int(field_modulus)
    rows = [[int(secret) % q for secret in secrets] for _ in range(int(committee_size))]
    coefficients = [
        [rng.randrange(q) for _ in secrets]
        for _ in range(int(threshold) - 1)
    ]
    for position in range(1, int(committee_size) + 1):
        power = 1
        row = rows[position - 1]
        for coefficient_row in coefficients:
            power = power * position % q
            for coordinate, coefficient in enumerate(coefficient_row):
                row[coordinate] = (row[coordinate] + power * coefficient) % q
    return tuple(tuple(row) for row in rows)


def reconstruct_shamir_vector(
    shares: Sequence[Sequence[int]],
    share_points: Sequence[int],
    field_modulus: int,
) -> tuple[int, ...]:
    if not shares:
        raise ValueError("at least one share is required")
    if len(shares) != len(share_points):
        raise ValueError("share count mismatch")
    width = len(shares[0])
    if any(len(row) != width for row in shares):
        raise ValueError("share vector width mismatch")
    q = int(field_modulus)
    coefficients = lagrange_coefficients_zero(share_points, q)
    return tuple(
        sum(coefficient * int(row[coordinate]) for coefficient, row in zip(coefficients, shares)) % q
        for coordinate in range(width)
    )


def lagrange_coefficients_zero(points: Sequence[int], field_modulus: int) -> tuple[int, ...]:
    q = int(field_modulus)
    normalized = tuple(int(value) % q for value in points)
    if len(set(normalized)) != len(normalized):
        raise ValueError("interpolation points must be unique")
    result: list[int] = []
    for index, point in enumerate(normalized):
        numerator = 1
        denominator = 1
        for other_index, other in enumerate(normalized):
            if index == other_index:
                continue
            numerator = numerator * (-other) % q
            denominator = denominator * (point - other) % q
        result.append(numerator * pow(denominator, -1, q) % q)
    return tuple(result)


def lagrange_evaluation_weights(
    points: Sequence[int],
    target: int,
    field_modulus: int,
) -> tuple[int, ...]:
    q = int(field_modulus)
    normalized = tuple(int(value) % q for value in points)
    target = int(target) % q
    if target in normalized:
        return tuple(1 if value == target else 0 for value in normalized)
    result: list[int] = []
    for index, point in enumerate(normalized):
        numerator = 1
        denominator = 1
        for other_index, other in enumerate(normalized):
            if index == other_index:
                continue
            numerator = numerator * (target - other) % q
            denominator = denominator * (point - other) % q
        result.append(numerator * pow(denominator, -1, q) % q)
    return tuple(result)


def polynomial_interpolate(
    points: Sequence[int],
    values: Sequence[int],
    field_modulus: int,
) -> tuple[int, ...]:
    if len(points) != len(values):
        raise ValueError("interpolation input length mismatch")
    q = int(field_modulus)
    result = [0 for _ in points]
    for index, point in enumerate(points):
        basis = [1]
        denominator = 1
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            basis = list(polynomial_multiply(basis, (-int(other), 1), q))
            denominator = denominator * (int(point) - int(other)) % q
        scale = int(values[index]) * pow(denominator, -1, q) % q
        for coefficient_index, coefficient in enumerate(basis):
            result[coefficient_index] = (result[coefficient_index] + scale * coefficient) % q
    return tuple(result)


def polynomial_multiply(
    left: Sequence[int],
    right: Sequence[int],
    field_modulus: int,
) -> tuple[int, ...]:
    q = int(field_modulus)
    result = [0 for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = (
                result[left_index + right_index]
                + int(left_value) * int(right_value)
            ) % q
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def polynomial_evaluate(
    coefficients: Sequence[int],
    point: int,
    field_modulus: int,
) -> int:
    q = int(field_modulus)
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * int(point) + int(coefficient)) % q
    return result


def parse_proof_share(values: Sequence[int], multiplication_count: int) -> PrioProofShare:
    expected = 2 * int(multiplication_count) + 6
    if len(values) != expected:
        raise ValueError("proof-share length mismatch")
    h_end = 2 + 2 * int(multiplication_count) + 1
    return PrioProofShare(
        f_zero=int(values[0]),
        g_zero=int(values[1]),
        h_coefficients=tuple(int(value) for value in values[2:h_end]),
        triple_a=int(values[h_end]),
        triple_b=int(values[h_end + 1]),
        triple_c=int(values[h_end + 2]),
    )


def dot_product(left: Sequence[int], right: Sequence[int], field_modulus: int) -> int:
    if len(left) != len(right):
        raise ValueError("dot-product length mismatch")
    q = int(field_modulus)
    return sum(int(a) * int(b) for a, b in zip(left, right)) % q


def integer_bits(value: int, bit_count: int) -> tuple[int, ...]:
    reduced = int(value) % (1 << int(bit_count))
    return tuple((reduced >> bit) & 1 for bit in range(int(bit_count)))


def random_bytes(rng: random.Random, length: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(int(length)))


def rejected_result(openings_valid: bool) -> PrioVerificationResult:
    return PrioVerificationResult(
        accepted=False,
        packet_openings_valid=bool(openings_valid),
        polynomial_identity_residual=None,
        constraint_residual=None,
        reconstructed_designated_output=None,
        adapter_output=None,
    )
