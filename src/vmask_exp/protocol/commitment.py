from __future__ import annotations

import secrets
from typing import Sequence


COMMITMENT_BYTES = 32
SALT_BYTES = 32
CHALLENGE_SEED_BYTES = 32
SESSION_ID_BYTES = 32
CLIENT_ID_BYTES = 8

def serialize_commitments(commitments: Sequence[bytes]) -> bytes:
    payload = bytearray()
    for commitment in commitments:
        if len(commitment) != COMMITMENT_BYTES:
            raise ValueError("invalid packet commitment length")
        payload.extend(commitment)
    return bytes(payload)


def generate_challenge_seed() -> bytes:
    return secrets.token_bytes(CHALLENGE_SEED_BYTES)
