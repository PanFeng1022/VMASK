from __future__ import annotations

import argparse
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import psutil

from vmask_exp.protocol.config import profile_p100, profile_p80
from vmask_exp.protocol.pipeline import VMaskProtocol
from vmask_exp.protocol.serialization import component_byte_counts


ACTIVE_CLIENTS = 2
MAX_CLIENTS = 20
MICROBENCHMARK_KAPPA = 2


class PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.002):
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.start_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start_bytes = self.process.memory_info().rss
        self.peak_bytes = self.start_bytes

        def sample() -> None:
            while not self._stop.is_set():
                self.peak_bytes = max(
                    self.peak_bytes,
                    self.process.memory_info().rss,
                )
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.start_bytes)


def protocol_config(profile: str):
    if profile == "P80":
        return profile_p80(
            committee_size=5,
            threshold=3,
            kappa=MICROBENCHMARK_KAPPA,
        )
    if profile == "P100":
        return profile_p100(
            committee_size=5,
            threshold=3,
            kappa=MICROBENCHMARK_KAPPA,
        )
    raise ValueError(f"unknown profile: {profile}")


def updates(dimension: int, repetition: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(((index + repetition) % 11) - 5 for index in range(dimension)),
        tuple(4 - ((2 * index + repetition) % 9) for index in range(dimension)),
    )


def run_iteration(profile: str, dimension: int, repetition: int) -> dict:
    cfg = protocol_config(profile)
    protocol = VMaskProtocol(
        cfg,
        dimension=dimension,
        max_clients=MAX_CLIENTS,
        committee_ids=(11, 22, 33, 44, 55),
        verification_ids=(44, 11, 33),
    )
    q_updates = updates(dimension, repetition)
    expected = tuple(
        first + second for first, second in zip(q_updates[0], q_updates[1])
    )
    with PeakRssSampler() as memory:
        result = protocol.execute_round(
            sid=f"native-full-snip-{profile.lower()}-{dimension}-{repetition}",
            q_updates=q_updates,
            seeds=(92000 + 2 * repetition, 92001 + 2 * repetition),
        )
    recovered = (
        tuple(int(value) for value in result.recovered_q_sum.detach().cpu().tolist())
        if hasattr(result.recovered_q_sum, "detach")
        else tuple(int(value) for value in result.recovered_q_sum)
    )
    if not result.released or not result.aggregate_equality or recovered != expected:
        raise RuntimeError("integrated native round failed correctness checks")
    timing = result.timing
    communication = component_byte_counts(cfg, dimension=dimension)
    prove_check = [
        prove + check
        for prove, check in zip(
            timing.client_prove_seconds,
            timing.client_check_seconds,
        )
    ]
    return {
        "repetition": repetition,
        "profile": profile,
        "dimension": dimension,
        "client_count": ACTIVE_CLIENTS,
        "max_clients": MAX_CLIENTS,
        "kappa": cfg.kappa,
        "committee_size": cfg.committee_size,
        "threshold": cfg.threshold,
        "prove_seconds_per_client": list(timing.client_prove_seconds),
        "prove_median_per_client_seconds": statistics.median(
            timing.client_prove_seconds
        ),
        "prove_serial_total_seconds": sum(timing.client_prove_seconds),
        "check_seconds_per_client": list(timing.client_check_seconds),
        "check_median_per_client_seconds": statistics.median(
            timing.client_check_seconds
        ),
        "check_serial_wall_total_seconds": sum(timing.client_check_seconds),
        "prove_check_seconds_per_client": prove_check,
        "prove_check_median_per_client_seconds": statistics.median(prove_check),
        "committee_serial_verification_seconds": (
            timing.committee_serial_verification_seconds
        ),
        "aggregate_recovery_seconds": timing.aggregate_recovery_seconds,
        "round_seconds": timing.round_seconds,
        "peak_rss_bytes": memory.peak_bytes,
        "peak_rss_delta_bytes": memory.peak_delta_bytes,
        "accepted_clients": result.accepted_clients,
        "rejected_clients": result.rejected_clients,
        "aggregate_equality": result.aggregate_equality,
        "backend_verification_material_bytes_per_client": (
            communication.backend_verification_material_bytes
        ),
        "implemented_client_payload_bytes": (
            communication.implemented_payload_subtotal_bytes
        ),
    }


def summarize(
    records: list[dict],
    field: str,
) -> dict[str, float | int | None]:
    values = sorted(float(record[field]) for record in records)
    result = {
        "sample_count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "q1": None,
        "q3": None,
        "iqr": None,
    }
    if len(values) > 1:
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        result.update(q1=q1, q3=q3, iqr=q3 - q1)
    return result


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("P80", "P100"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    arguments = parser.parse_args()

    if arguments.output.exists() and not arguments.append:
        parser.error(
            f"refusing to overwrite {arguments.output}; use --append to extend it"
        )

    for iteration in range(arguments.warmups):
        print(
            f"warmup profile={arguments.profile} "
            f"dimension={arguments.dimension} iteration={iteration}",
            flush=True,
        )
        run_iteration(arguments.profile, arguments.dimension, -iteration - 1)

    records = []
    if arguments.append and arguments.output.exists():
        previous = json.loads(arguments.output.read_text(encoding="utf-8"))
        if (
            previous.get("profile") != arguments.profile
            or int(previous.get("dimension", -1)) != arguments.dimension
            or any(
                int(record.get("kappa", -1)) != MICROBENCHMARK_KAPPA
                or int(record.get("max_clients", -1)) != MAX_CLIENTS
                for record in previous.get("records", ())
            )
        ):
            raise ValueError("append target uses a different benchmark configuration")
        records.extend(previous["records"])

    first_repetition = len(records)
    for repetition in range(
        first_repetition,
        first_repetition + arguments.repetitions,
    ):
        print(
            f"measure profile={arguments.profile} "
            f"dimension={arguments.dimension} repetition={repetition}",
            flush=True,
        )
        record = run_iteration(arguments.profile, arguments.dimension, repetition)
        records.append(record)
        print(json.dumps(record), flush=True)

    fields = (
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
    payload = {
        "scope": "native VMaskProtocol.execute_round with complete Threshold SNIP",
        "execution_boundary": {
            "active_client_count": ACTIVE_CLIENTS,
            "max_client_count": MAX_CLIENTS,
            "release_threshold_kappa": MICROBENCHMARK_KAPPA,
            "network_transport": False,
            "actual_distributed_parallel_execution": False,
            "prove": (
                "circuit and witness expansion, F/G/H construction, complete "
                "sharing, packet construction and commitment serialization"
            ),
            "check": (
                "selected packet opening checks, post-commitment challenges, "
                "three verifier computations executed serially, D/E opening, "
                "two residual reconstructions and designated-share check"
            ),
            "round": (
                "synthetic input and mask construction, two serial client Prove "
                "and Check executions, admission, aggregate mask recovery and "
                "aggregate correctness check"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "profile": arguments.profile,
        "dimension": arguments.dimension,
        "warmups": arguments.warmups,
        "repetitions": len(records),
        "added_repetitions": arguments.repetitions,
        "records": records,
        "summary": {field: summarize(records, field) for field in fields},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"OUTPUT={arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
