from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "prove_median_per_client_seconds",
    "prove_serial_total_seconds",
    "check_median_per_client_seconds",
    "check_serial_wall_total_seconds",
    "prove_check_median_per_client_seconds",
    "committee_serial_verification_seconds",
    "aggregate_recovery_seconds",
    "round_seconds",
    "peak_rss_bytes",
    "peak_rss_delta_bytes",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    arguments = parser.parse_args()
    rows = []
    for profile in ("P80", "P100"):
        paths = sorted(
            arguments.results_dir.glob(f"native_{profile}_d*.json"),
            key=lambda path: int(path.stem.rsplit("d", 1)[1]),
        )
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload["records"]
            dimension = int(payload["dimension"])
            row = {
                "profile": profile,
                "dimension": dimension,
                "client_count": records[0]["client_count"],
                "max_clients": records[0]["max_clients"],
                "kappa": records[0]["kappa"],
                "repetitions": payload["repetitions"],
                "all_rounds_correct": all(
                    record["aggregate_equality"]
                    and record["accepted_clients"] == [0, 1]
                    and record["rejected_clients"] == []
                    for record in records
                ),
                "backend_verification_material_bytes_per_client": records[0][
                    "backend_verification_material_bytes_per_client"
                ],
                "implemented_client_payload_bytes": records[0][
                    "implemented_client_payload_bytes"
                ],
            }
            for field in FIELDS:
                for statistic in ("median", "q1", "q3", "iqr", "min", "max"):
                    value = payload["summary"][field][statistic]
                    output_field = field
                    if value is not None and field.endswith("_bytes"):
                        value = value / (1024 * 1024)
                        output_field = field.removesuffix("_bytes") + "_mib"
                    row[f"{output_field}_{statistic}"] = value
            rows.append(row)

    output = arguments.results_dir / "native_full_snip_timing_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
