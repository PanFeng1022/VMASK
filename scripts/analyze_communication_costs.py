from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DIMENSIONS = (100, 1_000, 10_000, 27_562, 100_000, 105_050, 1_000_000)


def load_profiles():
    namespace = runpy.run_path(str(REPO / "src/vmask_exp/protocol/config.py"))
    return {"P80": namespace["profile_p80"](), "P100": namespace["profile_p100"]()}


def profile_cost(profile: str, cfg, d: int, n: int, tau: int, clients: int,
                 identity_bits: int, ring_bytes: int, field_bytes: int) -> dict:
    b_a = math.ceil(math.log2(2 * cfg.amax + 1))
    b_s = math.ceil(math.log2(2 * cfg.r0 + 1))
    g = d * (1 + 2 * b_a + 2 * cfg.limbs * b_s)
    p_snip = 3 * g + (2 * cfg.limbs + 1) * d + 6
    masked_update = d * ring_bytes
    mask_shares = n * d * field_bytes
    verification = tau * p_snip * field_bytes
    metadata = 77 * n + 32 * tau + 40
    client_payload = masked_update + mask_shares + verification + metadata
    residual_exchange = 6 * (tau - 1) * field_bytes
    recovery_ingress = (tau - 1) * d * field_bytes
    round_payload = (
        clients * (client_payload + identity_bits // 8 + residual_exchange)
        + recovery_ingress
    )
    if client_payload != sum((masked_update, mask_shares, verification, metadata)):
        raise AssertionError("component accounting does not sum to B_client")
    return {
        "profile": profile,
        "d": d,
        "n": n,
        "tau": tau,
        "N_t": clients,
        "b_A": b_a,
        "b_S": b_s,
        "L": cfg.limbs,
        "g": g,
        "p_snip": p_snip,
        "masked_update_bytes": masked_update,
        "mask_shares_bytes": mask_shares,
        "verification_bytes": verification,
        "metadata_bytes": metadata,
        "B_client_bytes": client_payload,
        "B_round_bytes": round_payload,
        "recovery_ingress_bytes": recovery_ingress,
    }


def lightsecagg_recovery_bytes(d: int, users: int = 14, privacy: int = 2) -> int:
    return 4 * users * math.ceil(d / (users - privacy))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--committee-size", type=int, default=5)
    parser.add_argument("--thresholds", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--identity-bits", type=int, default=256)
    args = parser.parse_args()
    if any(d <= 0 for d in args.dimensions):
        parser.error("all dimensions must be positive")
    if args.clients <= 0 or args.committee_size < 2 or args.identity_bits <= 0:
        parser.error("clients and identity-bits must be positive; committee-size must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    rows = []
    for profile, cfg in profiles.items():
        for tau in args.thresholds:
            if not 2 <= tau <= args.committee_size:
                raise ValueError("each threshold must satisfy 2 <= tau <= n")
            for d in args.dimensions:
                rows.append(profile_cost(profile, cfg, d, args.committee_size, tau,
                                         args.clients, args.identity_bits, 16, 24))
    csv_path = output / "communication_costs.csv"
    json_path = output / "communication_costs.json"
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    write_csv(csv_path, rows)
    payload = {
        "classification": "exact analytical application-format payload; not network measurement",
        "units": "MB=10^6 bytes; GB=10^9 bytes",
        "formulas": {
            "g": "d(1+2b_A+2Lb_S)",
            "p_snip": "3g+(2L+1)d+6",
            "B_ver": "tau*p_snip*b_q",
            "B_client": "d(b_M+n*b_q)+B_ver+77n+32tau+40",
            "B_round": "N_t[B_client+lambda_iota/8+6(tau-1)b_q]+(tau-1)d*b_q",
            "VMASK recovery": "(tau-1)d*b_q",
            "LightSecAgg recovery": "4U*ceil(d/(U-T)), (U,T)=(14,2)",
        },
        "element_bytes": {"b_M": 16, "b_q": 24, "LightSecAgg": 4},
        "rows": rows,
        "lightsecagg_recovery": [
            {"d": d, "bytes": lightsecagg_recovery_bytes(d)} for d in args.dimensions],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE={csv_path}")
    print(f"WROTE={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
