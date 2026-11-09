from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .commitment import (
    CHALLENGE_SEED_BYTES,
    CLIENT_ID_BYTES,
    COMMITMENT_BYTES,
    SALT_BYTES,
    SESSION_ID_BYTES,
    serialize_commitments,
)
from .config import ProtocolConfig
from .pipeline import ClientSubmission, ProtectedSubmission
from .prio_snip_reference import (
    PrioPacket,
    cost_estimate,
    serialize_packet_payload,
)
from .threshold_snip import BACKEND_NAME


AGGREGATION_ELEMENT_BYTES = 16
FIELD_ELEMENT_BYTES = 24


@dataclass(frozen=True)
class CommunicationComponents:
    masked_update_bytes: int
    designated_share_bytes: int
    backend_verification_material_bytes: int
    packet_framing_bytes: int
    commitment_control_bytes: int
    commitment_bytes: int
    salt_bytes: int
    metadata_bytes: int
    vmask_specific_client_payload_bytes: int
    implemented_payload_subtotal_bytes: int
    challenge_seed_bytes: int
    d_e_opening_bytes: int
    residual_reconstruction_bytes: int
    aggregate_recovery_communication_bytes: int


def encode_aggregation_element(value: int, cfg: ProtocolConfig) -> bytes:
    return _encode_mod(value, cfg.modulus, AGGREGATION_ELEMENT_BYTES)


def decode_aggregation_element(data: bytes, cfg: ProtocolConfig) -> int:
    return _decode_mod(data, cfg.modulus, AGGREGATION_ELEMENT_BYTES)


def encode_field_element(value: int, cfg: ProtocolConfig) -> bytes:
    return _encode_mod(value, cfg.field_modulus, FIELD_ELEMENT_BYTES)


def decode_field_element(data: bytes, cfg: ProtocolConfig) -> int:
    return _decode_mod(data, cfg.field_modulus, FIELD_ELEMENT_BYTES)


def serialize_masked_update(submission: ProtectedSubmission, cfg: ProtocolConfig) -> bytes:
    return serialize_aggregation_vector(submission.y, cfg)


def deserialize_masked_update(data: bytes, cfg: ProtocolConfig) -> list[int]:
    return deserialize_aggregation_vector(data, cfg)


def serialize_designated_shares(
    packets: Sequence[PrioPacket],
    cfg: ProtocolConfig,
) -> bytes:
    payload = bytearray()
    for packet in packets:
        payload.extend(serialize_field_vector(packet.designated_shamir_share, cfg))
    return bytes(payload)


def deserialize_designated_shares(
    data: bytes,
    cfg: ProtocolConfig,
    *,
    committee_size: int,
    dimension: int,
) -> list[list[int]]:
    expected = committee_size * dimension * FIELD_ELEMENT_BYTES
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(data)}")
    rows: list[list[int]] = []
    cursor = 0
    row_bytes = dimension * FIELD_ELEMENT_BYTES
    for _ in range(committee_size):
        rows.append(deserialize_field_vector(data[cursor : cursor + row_bytes], cfg))
        cursor += row_bytes
    return rows


def serialize_backend_verification_material(
    packets: Sequence[PrioPacket],
    cfg: ProtocolConfig,
) -> bytes:

    payload = bytearray()
    for packet in packets:
        if packet.non_designated_additive_share is None:
            if packet.proof_share is not None:
                raise ValueError("nonselected packet contains a proof share")
            continue
        if packet.proof_share is None:
            raise ValueError("selected packet lacks a proof share")
        payload.extend(
            serialize_field_vector(packet.non_designated_additive_share, cfg)
        )
        proof = packet.proof_share
        payload.extend(
            serialize_field_vector((proof.f_zero, proof.g_zero), cfg)
        )
        payload.extend(serialize_field_vector(proof.h_coefficients, cfg))
        payload.extend(
            serialize_field_vector(
                (proof.triple_a, proof.triple_b, proof.triple_c),
                cfg,
            )
        )
    return bytes(payload)


def serialize_commitment_control(
    packets: Sequence[PrioPacket],
) -> bytes:
    payload = bytearray()
    for packet in packets:
        if len(packet.commitment_randomness) != SALT_BYTES:
            raise ValueError("invalid packet salt length")
        payload.extend(packet.commitment_randomness)
    return bytes(payload)


def deserialize_commitment_control(
    data: bytes,
    *,
    committee_size: int,
) -> list[bytes]:
    expected = int(committee_size) * SALT_BYTES
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(data)}")
    return [
        data[offset : offset + SALT_BYTES]
        for offset in range(0, len(data), SALT_BYTES)
    ]


def serialize_public_commitments(submission: ClientSubmission) -> bytes:
    return serialize_commitments(submission.evidence.packet_commitments)


def deserialize_public_commitments(data: bytes, *, committee_size: int) -> tuple[bytes, ...]:
    expected = int(committee_size) * COMMITMENT_BYTES
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(data)}")
    return tuple(
        data[offset : offset + COMMITMENT_BYTES]
        for offset in range(0, len(data), COMMITMENT_BYTES)
    )


def serialize_submission_metadata(submission: ClientSubmission) -> bytes:
    sid = hashlib.sha256(submission.gamma.sid.encode("utf-8")).digest()
    client_id = int(submission.gamma.client_id).to_bytes(
        CLIENT_ID_BYTES,
        "big",
        signed=False,
    )
    return sid + client_id + serialize_public_commitments(submission)


def serialize_aggregate_recovery_shares(
    shares: Sequence[Sequence[int]],
    cfg: ProtocolConfig,
) -> bytes:
    payload = bytearray()
    for share in shares:
        payload.extend(serialize_field_vector(share, cfg))
    return bytes(payload)


def deserialize_aggregate_recovery_shares(
    data: bytes,
    cfg: ProtocolConfig,
    *,
    share_count: int,
    dimension: int,
) -> list[list[int]]:
    expected = share_count * dimension * FIELD_ELEMENT_BYTES
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(data)}")
    rows: list[list[int]] = []
    cursor = 0
    row_bytes = dimension * FIELD_ELEMENT_BYTES
    for _ in range(share_count):
        rows.append(deserialize_field_vector(data[cursor : cursor + row_bytes], cfg))
        cursor += row_bytes
    return rows


def serialize_aggregation_vector(values: Sequence[int], cfg: ProtocolConfig) -> bytes:
    return b"".join(encode_aggregation_element(value, cfg) for value in values)


def deserialize_aggregation_vector(data: bytes, cfg: ProtocolConfig) -> list[int]:
    if len(data) % AGGREGATION_ELEMENT_BYTES != 0:
        raise ValueError("aggregation vector payload has invalid length")
    return [
        decode_aggregation_element(data[offset : offset + AGGREGATION_ELEMENT_BYTES], cfg)
        for offset in range(0, len(data), AGGREGATION_ELEMENT_BYTES)
    ]


def serialize_field_vector(values: Sequence[int], cfg: ProtocolConfig) -> bytes:
    return b"".join(encode_field_element(value, cfg) for value in values)


def deserialize_field_vector(data: bytes, cfg: ProtocolConfig) -> list[int]:
    if len(data) % FIELD_ELEMENT_BYTES != 0:
        raise ValueError("field vector payload has invalid length")
    return [
        decode_field_element(data[offset : offset + FIELD_ELEMENT_BYTES], cfg)
        for offset in range(0, len(data), FIELD_ELEMENT_BYTES)
    ]


def component_byte_counts(
    cfg: ProtocolConfig,
    *,
    dimension: int,
) -> CommunicationComponents:
    masked_update = int(dimension) * AGGREGATION_ELEMENT_BYTES
    designated_shares = cfg.committee_size * int(dimension) * FIELD_ELEMENT_BYTES
    estimate = cost_estimate(cfg, int(dimension))
    backend_verification_material = (
        estimate.total_client_field_elements - cfg.committee_size * int(dimension)
    ) * FIELD_ELEMENT_BYTES
    packet_framing = (
        cfg.threshold * 45
        + (cfg.committee_size - cfg.threshold) * 13
    )
    commitment = cfg.committee_size * COMMITMENT_BYTES
    salt = cfg.committee_size * SALT_BYTES
    commitment_control = cfg.committee_size * SALT_BYTES
    metadata = SESSION_ID_BYTES + CLIENT_ID_BYTES + commitment
    vmask_specific = masked_update + designated_shares
    implemented_subtotal = (
        vmask_specific
        + backend_verification_material
        + packet_framing
        + commitment_control
        + metadata
    )
    d_e_opening = 4 * (cfg.threshold - 1) * FIELD_ELEMENT_BYTES
    residual_reconstruction = 2 * (cfg.threshold - 1) * FIELD_ELEMENT_BYTES
    aggregate_recovery = (cfg.threshold - 1) * int(dimension) * FIELD_ELEMENT_BYTES
    return CommunicationComponents(
        masked_update_bytes=masked_update,
        designated_share_bytes=designated_shares,
        backend_verification_material_bytes=backend_verification_material,
        packet_framing_bytes=packet_framing,
        commitment_control_bytes=commitment_control,
        commitment_bytes=commitment,
        salt_bytes=salt,
        metadata_bytes=metadata,
        vmask_specific_client_payload_bytes=vmask_specific,
        implemented_payload_subtotal_bytes=implemented_subtotal,
        challenge_seed_bytes=CHALLENGE_SEED_BYTES,
        d_e_opening_bytes=d_e_opening,
        residual_reconstruction_bytes=residual_reconstruction,
        aggregate_recovery_communication_bytes=aggregate_recovery,
    )


def measure_submission_components(
    submission: ClientSubmission,
    cfg: ProtocolConfig,
) -> CommunicationComponents:
    dimension = len(submission.gamma.y)
    masked_update = len(serialize_masked_update(submission.gamma, cfg))
    designated_shares = len(
        serialize_designated_shares(submission.evidence.verifier_packets, cfg)
    )
    if submission.evidence.backend != BACKEND_NAME:
        raise ValueError("submission does not use the complete Threshold SNIP backend")
    if not all(
        isinstance(packet, PrioPacket)
        for packet in submission.evidence.verifier_packets
    ):
        raise TypeError("complete backend requires complete Prio packets")
    backend_verification_material = len(
        serialize_backend_verification_material(
            submission.evidence.verifier_packets,
            cfg,
        )
    )
    serialized_payload_bytes = sum(
        len(serialize_packet_payload(packet, cfg.field_modulus))
        for packet in submission.evidence.verifier_packets
    )
    packet_framing = (
        serialized_payload_bytes
        - designated_shares
        - backend_verification_material
    )
    commitment_control = len(
        serialize_commitment_control(submission.evidence.verifier_packets)
    )
    commitment = len(serialize_public_commitments(submission))
    salt = sum(
        len(packet.commitment_randomness)
        for packet in submission.evidence.verifier_packets
    )
    metadata = len(serialize_submission_metadata(submission))
    vmask_specific = masked_update + designated_shares
    implemented_subtotal = (
        vmask_specific
        + backend_verification_material
        + packet_framing
        + commitment_control
        + metadata
    )
    d_e_opening = 4 * (cfg.threshold - 1) * FIELD_ELEMENT_BYTES
    residual_reconstruction = 2 * (cfg.threshold - 1) * FIELD_ELEMENT_BYTES
    aggregate_recovery = (cfg.threshold - 1) * dimension * FIELD_ELEMENT_BYTES
    return CommunicationComponents(
        masked_update_bytes=masked_update,
        designated_share_bytes=designated_shares,
        backend_verification_material_bytes=backend_verification_material,
        packet_framing_bytes=packet_framing,
        commitment_control_bytes=commitment_control,
        commitment_bytes=commitment,
        salt_bytes=salt,
        metadata_bytes=metadata,
        vmask_specific_client_payload_bytes=vmask_specific,
        implemented_payload_subtotal_bytes=implemented_subtotal,
        challenge_seed_bytes=CHALLENGE_SEED_BYTES,
        d_e_opening_bytes=d_e_opening,
        residual_reconstruction_bytes=residual_reconstruction,
        aggregate_recovery_communication_bytes=aggregate_recovery,
    )


def _encode_mod(value: int, modulus: int, width: int) -> bytes:
    encoded = int(value) % int(modulus)
    if encoded >= 1 << (8 * width):
        raise ValueError("encoded value does not fit fixed-width serialization")
    return encoded.to_bytes(width, "big", signed=False)


def _decode_mod(data: bytes, modulus: int, width: int) -> int:
    if len(data) != width:
        raise ValueError(f"expected {width} bytes, got {len(data)}")
    value = int.from_bytes(data, "big", signed=False)
    if value >= int(modulus):
        raise ValueError("serialized value is outside the target domain")
    return value
