from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vmask_exp.protocol import threshold_snip
from vmask_exp.protocol.config import profile_p100, profile_p80
from vmask_exp.protocol.pipeline import VMaskProtocol


class PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.002):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.start_bytes = 0
        self.peak_bytes = 0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.start_bytes = self.process.memory_info().rss
        self.peak_bytes = self.start_bytes

        def sample():
            while not self.stop.is_set():
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
                self.stop.wait(self.interval_seconds)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def source_hashes() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "src/vmask_exp/protocol").glob("*.py"))
    }


def committee_configuration(n: int, tau: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    committee = tuple(11 * (index + 1) for index in range(n))
    preferred = (44, 11, 33, 22, 55, 66, 77, 88, 99)
    selected = tuple(value for value in preferred if value in committee)[:tau]
    if len(selected) != tau:
        raise ValueError("committee size exceeds the supported deterministic ID pool")
    return committee, selected


def make_config(args: argparse.Namespace):
    factory = profile_p80 if args.profile == "P80" else profile_p100
    return factory(
        committee_size=args.committee_size,
        threshold=args.threshold,
        kappa=args.kappa,
    )


def updates(client_count: int, dimension: int, repetition: int):
    return tuple(
        tuple(((index * (client_id + 1) + client_id + repetition) % 11) - 5
              for index in range(dimension))
        for client_id in range(client_count)
    )


def as_tuple(value) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return tuple(int(item) for item in value)


def run_once(args: argparse.Namespace, repetition: int) -> dict:
    cfg = make_config(args)
    committee, selected = committee_configuration(args.committee_size, args.threshold)
    protocol = VMaskProtocol(
        cfg,
        dimension=args.dimension,
        max_clients=args.max_clients,
        committee_ids=committee,
        verification_ids=selected,
    )
    q_updates = updates(args.clients, args.dimension, repetition)
    seeds = tuple(args.seed_base + 1000 * repetition + client_id
                  for client_id in range(args.clients))
    expected = tuple(sum(row[index] for row in q_updates)
                     for index in range(args.dimension))
    with PeakRssSampler() as memory:
        result = protocol.execute_round(
            sid=(f"native-scalability-{args.profile.lower()}-d{args.dimension}-"
                 f"N{args.clients}-n{args.committee_size}-t{args.threshold}-r{repetition}"),
            q_updates=q_updates,
            seeds=seeds,
        )
    recovered = as_tuple(result.recovered_q_sum)
    if not result.released:
        raise RuntimeError(f"unexpected abort: {result.release_reason}")
    if not result.aggregate_equality or recovered != expected:
        raise RuntimeError("released aggregate does not equal the independent sum")
    if result.accepted_clients != list(range(args.clients)) or result.rejected_clients:
        raise RuntimeError("benign scalability run changed the accepted client set")
    timing = result.timing
    prove_check = [a + b for a, b in zip(
        timing.client_prove_seconds, timing.client_check_seconds)]
    return {
        "repetition": repetition,
        "profile": args.profile,
        "dimension": args.dimension,
        "client_count": args.clients,
        "max_clients": args.max_clients,
        "kappa": args.kappa,
        "committee_ids": list(committee),
        "verification_ids": list(selected),
        "committee_size": args.committee_size,
        "threshold": args.threshold,
        "prove_seconds_per_client": list(timing.client_prove_seconds),
        "prove_median_per_client_seconds": statistics.median(timing.client_prove_seconds),
        "prove_serial_total_seconds": sum(timing.client_prove_seconds),
        "check_seconds_per_client": list(timing.client_check_seconds),
        "check_median_per_client_seconds": statistics.median(timing.client_check_seconds),
        "check_serial_total_seconds": sum(timing.client_check_seconds),
        "prove_check_median_per_client_seconds": statistics.median(prove_check),
        "committee_serial_verification_seconds": timing.committee_serial_verification_seconds,
        "aggregate_recovery_seconds": timing.aggregate_recovery_seconds,
        "round_seconds": timing.round_seconds,
        "peak_rss_bytes": memory.peak_bytes,
        "peak_rss_delta_bytes": max(0, memory.peak_bytes - memory.start_bytes),
        "accepted_clients": result.accepted_clients,
        "rejected_clients": result.rejected_clients,
        "aggregate_equality": result.aggregate_equality,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(records: list[dict], field: str) -> dict[str, float | int]:
    values = [float(record[field]) for record in records]
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    return {
        "sample_count": len(values),
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("P80", "P100"), required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--max-clients", type=int, default=20)
    parser.add_argument("--committee-size", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--kappa", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=2_900_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.clients <= args.max_clients):
        parser.error("clients must satisfy 1 <= clients <= max-clients")
    if not (1 <= args.kappa <= args.clients):
        parser.error("kappa must satisfy 1 <= kappa <= clients")
    if not (2 <= args.threshold <= args.committee_size):
        parser.error("threshold must satisfy 2 <= threshold <= committee-size")
    if args.dimension <= 0 or args.repetitions <= 0 or args.warmups < 0:
        parser.error("dimension/repetitions must be positive and warmups nonnegative")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    cfg = make_config(args)
    cfg.validate(args.dimension, args.max_clients)
    b_a = (2 * cfg.amax + 1).bit_length()
    b_s = (2 * cfg.r0 + 1).bit_length()
    multiplication_count = args.dimension * (1 + 2 * b_a + 2 * cfg.limbs * b_s)
    before = source_hashes()
    setup_start = time.perf_counter()
    setup = threshold_snip.prepare_public_interpolation(
        cfg.field_modulus, multiplication_count)
    setup_seconds = time.perf_counter() - setup_start
    for warmup in range(args.warmups):
        run_once(args, -(warmup + 1))
    records = []
    for repetition in range(args.repetitions):
        record = run_once(args, repetition)
        records.append(record)
        print(json.dumps(record), flush=True)
    after = source_hashes()
    if before != after:
        raise RuntimeError("protocol source files changed during the benchmark")
    fields = (
        "prove_median_per_client_seconds", "prove_serial_total_seconds",
        "check_median_per_client_seconds", "check_serial_total_seconds",
        "prove_check_median_per_client_seconds",
        "committee_serial_verification_seconds", "aggregate_recovery_seconds",
        "round_seconds", "peak_rss_bytes", "peak_rss_delta_bytes",
    )
    payload = {
        "scope": "complete native Threshold SNIP local execution",
        "boundary": {
            "includes": ["Prove", "selected-committee Check", "admission",
                         "aggregate mask recovery", "aggregate correctness check"],
            "excludes": ["network transport", "distributed parallel execution"],
        },
        "backend": threshold_snip.BACKEND_NAME,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "psutil": psutil.__version__,
        },
        "configuration": vars(args) | {"output": args.output.name},
        "public_preprocessing": {
            "seconds": setup_seconds,
            "multiplication_count": multiplication_count,
            "point_count": setup.point_count,
        },
        "records": records,
        "summary": {field: summarize(records, field) for field in fields},
        "source_hashes": before,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
