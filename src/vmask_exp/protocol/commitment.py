from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import ProtocolConfig
from .snip import SnipStatement


COMMITMENT_BYTES = 32
SALT_BYTES = 32
CHALLENGE_SEED_BYTES = 32
SESSION_ID_BYTES = 32
CLIENT_ID_BYTES = 8


@dataclass(frozen=True)
class PacketCommitmentBundle:
    statement_digest: bytes
    commitments: tuple[bytes, ...]
    salts: tuple[bytes, ...]


def serialize_statement(statement: SnipStatement, cfg: ProtocolConfig) -> bytes:
    payload = bytearray()
    payload.extend(bytes.fromhex(statement.theta_digest))
    payload.extend(hashlib.sha256(str(statement.sid).encode("utf-8")).digest())
    payload.extend(_encode_uint32(len(statement.committee_ids)))
    for committee_id in statement.committee_ids:
        payload.extend(_encode_uint32(committee_id))
    payload.extend(int(statement.client_id).to_bytes(CLIENT_ID_BYTES, "big", signed=False))
    payload.extend(_encode_uint32(len(statement.y)))
    for value in statement.y:
        payload.extend(_encode_mod(value, cfg.modulus, 16))
    return bytes(payload)


def statement_digest(statement: SnipStatement, cfg: ProtocolConfig) -> bytes:
    return hashlib.sha256(
        b"VMASK-STMT-v1" + serialize_statement(statement, cfg)
    ).digest()


def serialize_packet_payload(
    z_shares: Sequence[int],
    verification_share: int,
    cfg: ProtocolConfig,
) -> bytes:
    payload = bytearray()
    for value in z_shares:
        payload.extend(_encode_mod(value, cfg.field_modulus, 24))
    payload.extend(_encode_mod(verification_share, cfg.field_modulus, 24))
    return bytes(payload)


def generate_packet_commitments(
    statement: SnipStatement,
    packet_payloads: Sequence[tuple[Sequence[int], int]],
    cfg: ProtocolConfig,
    *,
    salt_source: Callable[[int], bytes] = secrets.token_bytes,
) -> PacketCommitmentBundle:
    if len(packet_payloads) != cfg.committee_size:
        raise ValueError("packet count does not match committee_size")
    stmt_digest = statement_digest(statement, cfg)
    commitments: list[bytes] = []
    salts: list[bytes] = []
    for h, (z_shares, verification_share) in enumerate(packet_payloads, start=1):
        salt = salt_source(SALT_BYTES)
        if len(salt) != SALT_BYTES:
            raise ValueError("salt source returned the wrong number of bytes")
        payload = serialize_packet_payload(z_shares, verification_share, cfg)
        commitment = hashlib.sha256(
            b"VMASK-PACKET-v1"
            + stmt_digest
            + _encode_uint32(h)
            + salt
            + payload
        ).digest()
        salts.append(salt)
        commitments.append(commitment)
    return PacketCommitmentBundle(
        statement_digest=stmt_digest,
        commitments=tuple(commitments),
        salts=tuple(salts),
    )


def verify_packet_commitment(
    *,
    statement: SnipStatement,
    committee_position: int,
    z_shares: Sequence[int],
    verification_share: int,
    salt: bytes,
    commitment: bytes,
    cfg: ProtocolConfig,
) -> bool:
    if not 1 <= int(committee_position) <= cfg.committee_size:
        return False
    if len(salt) != SALT_BYTES or len(commitment) != COMMITMENT_BYTES:
        return False
    stmt_digest = statement_digest(statement, cfg)
    payload = serialize_packet_payload(z_shares, verification_share, cfg)
    expected = hashlib.sha256(
        b"VMASK-PACKET-v1"
        + stmt_digest
        + _encode_uint32(committee_position)
        + salt
        + payload
    ).digest()
    return secrets.compare_digest(expected, commitment)


def serialize_commitments(commitments: Sequence[bytes]) -> bytes:
    payload = bytearray()
    for commitment in commitments:
        if len(commitment) != COMMITMENT_BYTES:
            raise ValueError("invalid packet commitment length")
        payload.extend(commitment)
    return bytes(payload)


def generate_challenge_seed() -> bytes:
    return secrets.token_bytes(CHALLENGE_SEED_BYTES)


def derive_challenge(
    *,
    statement: SnipStatement,
    commitments: Sequence[bytes],
    chi: bytes,
    cfg: ProtocolConfig,
) -> int:
    if len(chi) != CHALLENGE_SEED_BYTES:
        raise ValueError("challenge seed must contain 32 bytes")
    material = (
        b"VMASK-CHALLENGE-v1"
        + statement_digest(statement, cfg)
        + serialize_commitments(commitments)
        + chi
    )
    return hash_to_field(material, cfg.field_modulus)


def hash_to_field(material: bytes, modulus: int) -> int:
    modulus = int(modulus)
    if modulus <= 1:
        raise ValueError("field modulus must exceed one")
    width = (modulus.bit_length() + 7) // 8
    if width > hashlib.sha256().digest_size:
        raise ValueError("SHA-256 output is too short for this field modulus")
    excess_bits = 8 * width - modulus.bit_length()
    for counter in range(1 << 32):
        digest = hashlib.sha256(material + _encode_uint32(counter)).digest()
        candidate = int.from_bytes(digest[:width], "big", signed=False)
        if excess_bits:
            candidate >>= excess_bits
        if candidate < modulus:
            return candidate
    raise RuntimeError("hash-to-field rejection sampling exhausted its counter space")


def _encode_uint32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


def _encode_mod(value: int, modulus: int, width: int) -> bytes:
    encoded = int(value) % int(modulus)
    if encoded >= 1 << (8 * width):
        raise ValueError("encoded value does not fit fixed-width serialization")
    return encoded.to_bytes(width, "big", signed=False)
