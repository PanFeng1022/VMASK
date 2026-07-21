from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmask_exp.protocol import (
    AGGREGATION_ELEMENT_BYTES,
    FIELD_ELEMENT_BYTES,
    SnipStatement,
    SnipWitness,
    VMaskProtocol,
    VMaskSnip,
    component_byte_counts,
    protocol_parameter_digest,
)
from vmask_exp.protocol.pipeline import CommitteeMaterial, ProtectedSubmission, VerificationEvidence
from vmask_exp.protocol.commitment import (
    generate_packet_commitments,
    verify_packet_commitment,
)
from vmask_exp.protocol.config import profile_p100, profile_p80


Q_SNIP = (1 << 192) - (1 << 64) - 1
N_MAX = 20
COMMITTEE_N = 5
THRESHOLD = 3
FIELD_BYTES = FIELD_ELEMENT_BYTES
MASKED_UPDATE_BYTES = AGGREGATION_ELEMENT_BYTES
EPS = 1e-12


@dataclass(frozen=True)
class Profile:
    name: str
    r0: int
    limbs: int
    beta: int
    amax: int = 15

    @property
    def b_a(self) -> int:
        return math.ceil(math.log2(2 * self.amax + 1))

    @property
    def b_s(self) -> int:
        return math.ceil(math.log2(2 * self.r0 + 1))


PROFILES = [
    Profile(name="P80", r0=1023, limbs=10, beta=2047),
    Profile(name="P100", r0=8191, limbs=9, beta=16383),
]

MODELS = [
    ("MNIST", 27_562),
    ("CIFAR-10", 105_050),
]

SCALABILITY_DIMS = [1_000, 10_000, 27_562, 105_050]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def median_iqr(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), percentile(values, 0.75) - percentile(values, 0.25)


def fmt_sec(value: float) -> str:
    return f"{value:.3f}"


def fmt_mib(value: float) -> str:
    return f"{value:.3f}"


def workload(profile: Profile, dim: int) -> dict[str, int]:
    mult_gates = dim * (1 + 2 * profile.b_a + 2 * profile.limbs * profile.b_s)
    return {"Gx": mult_gates, "Nout": dim}


def protocol_config(profile: Profile):
    if profile.name == "P80":
        return profile_p80(committee_size=COMMITTEE_N, threshold=THRESHOLD)
    if profile.name == "P100":
        return profile_p100(committee_size=COMMITTEE_N, threshold=THRESHOLD)
    raise ValueError(f"unsupported profile: {profile.name}")


def build_snip_instance(profile: Profile, dim: int, seed: int):
    cfg = protocol_config(profile)
    theta_digest = protocol_parameter_digest(
        cfg,
        dimension=dim,
        max_clients=N_MAX,
    )
    committee_ids = tuple(range(1, cfg.committee_size + 1))
    snip = VMaskSnip(
        cfg,
        dim,
        theta_digest=theta_digest,
        committee_ids=committee_ids,
        verification_ids=committee_ids[: cfg.threshold],
    )
    rng = np.random.default_rng(seed)
    x = [int(v) for v in rng.integers(-profile.amax, profile.amax + 1, size=dim, dtype=np.int64)]
    limbs = [
        [int(v) for v in rng.integers(-profile.r0, profile.r0 + 1, size=dim, dtype=np.int64)]
        for _ in range(profile.limbs)
    ]
    mask = snip.compose_mask(limbs)
    y_source = [x[r] + mask[r] for r in range(dim)]
    delta = [1 if value < 0 else 0 for value in y_source]
    y = [value + delta[r] * cfg.modulus for r, value in enumerate(y_source)]
    shifted = [[limbs[ell][r] + profile.r0 for r in range(dim)] for ell in range(profile.limbs)]
    z = snip.compose_mask(limbs)
    statement = SnipStatement(
        theta_digest=theta_digest,
        sid=f"bench-{profile.name}-{dim}",
        committee_ids=committee_ids[: cfg.threshold],
        client_id=0,
        y=y,
    )
    witness = SnipWitness(x=x, limbs=limbs, shifted=shifted, delta=delta, z=z)
    return snip, statement, witness, x


def pin_current_process(cpu_index: int | None) -> None:
    if cpu_index is None:
        return
    if cpu_index < 0:
        raise ValueError("affinity CPU index must be nonnegative")
    mask = 1 << cpu_index
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu_index})
    else:
        raise RuntimeError("CPU affinity is not supported on this platform")


def calibration_samples(repeats: int, samples: int, warmup_runs: int = 0) -> list[float]:
    values: list[float] = []
    for rep in range(repeats + warmup_runs):
        gc.collect()
        a = 123456789012345678901234567890123456789 + rep
        b = 987654321098765432109876543210987654321 + rep
        c = 1442695040888963407
        start = time.perf_counter()
        for i in range(samples):
            a = (a * b + c + i) % Q_SNIP
            b = (b + a + c) % Q_SNIP
        elapsed = time.perf_counter() - start
        if rep >= warmup_runs:
            values.append(elapsed / samples)
    return values


def bench_client_mask_composition(profile: Profile, dim: int, repeats: int) -> list[float]:
    times: list[float] = []
    rng = np.random.default_rng(20260705 + dim + profile.r0)
    for _ in range(repeats):
        start = time.perf_counter()
        update = rng.integers(-profile.amax, profile.amax + 1, size=dim, dtype=np.int64)
        limbs = rng.integers(-profile.r0, profile.r0 + 1, size=(profile.limbs, dim), dtype=np.int64)
        mask = np.zeros(dim, dtype=object)
        for ell in range(profile.limbs):
            mask = mask + limbs[ell].astype(object) * (profile.beta**ell)
        masked = update.astype(object) + mask
        shifted = limbs + profile.r0
        # Consume values so NumPy cannot elide the computation.
        _ = int(masked[0]) ^ int(shifted[0, 0])
        times.append(time.perf_counter() - start)
    return times


def bench_aggregate_recovery(
    profile: Profile,
    dim: int,
    repeats: int,
    warmup_runs: int = 0,
) -> list[float]:
    times: list[float] = []
    rng = np.random.default_rng(20260705 + 7 * dim + profile.r0)
    masked_updates = rng.integers(-profile.amax, profile.amax + 1, size=(N_MAX, dim), dtype=np.int64)
    share_values = rng.integers(
        -profile.r0,
        profile.r0 + 1,
        size=(N_MAX, dim),
        dtype=np.int64,
    )
    lagrange = np.asarray([3, -3, 1], dtype=np.int64)
    committee_shares = rng.integers(
        -profile.r0,
        profile.r0 + 1,
        size=(THRESHOLD, dim),
        dtype=np.int64,
    )
    for rep in range(repeats + warmup_runs):
        gc.collect()
        start = time.perf_counter()
        y_agg = masked_updates.sum(axis=0)
        certified = share_values.sum(axis=0)
        reconstructed = (
            lagrange[0] * committee_shares[0]
            + lagrange[1] * committee_shares[1]
            + lagrange[2] * committee_shares[2]
            + certified
        )
        recovered = y_agg - reconstructed
        _ = int(recovered[0])
        elapsed = time.perf_counter() - start
        if rep >= warmup_runs:
            times.append(elapsed)
    return times


def bench_snip_proving(profile: Profile, dim: int, repeats: int) -> list[float]:
    times: list[float] = []
    snip, statement, witness, _ = build_snip_instance(profile, dim, 20260706 + dim + profile.r0)
    for rep in range(repeats):
        start = time.perf_counter()
        proof = snip.prove(statement, witness, seed=20260706 + rep)
        if not proof.constraints.satisfied:
            raise AssertionError("valid SNIP witness failed during cost benchmark")
        times.append(time.perf_counter() - start)
    return times


def bench_client_submission(profile: Profile, dim: int, repeats: int) -> list[float]:
    times: list[float] = []
    cfg = protocol_config(profile)
    protocol = VMaskProtocol(cfg, dimension=dim, max_clients=N_MAX)
    _, _, _, q_update = build_snip_instance(profile, dim, 20260706 + 3 * dim + profile.r0)
    for rep in range(repeats):
        start = time.perf_counter()
        submission = protocol.make_submission(
            sid=f"bench-client-{profile.name}-{dim}-{rep}",
            client_id=0,
            q_update=q_update,
            seed=20260706 + 10_000 + rep,
        )
        if not submission.accepted_by_relation:
            raise AssertionError("valid client submission failed during cost benchmark")
        times.append(time.perf_counter() - start)
    return times


def bench_client_submission_breakdown(
    profile: Profile,
    dim: int,
    repeats: int,
    warmup_runs: int = 0,
) -> dict[str, list[float]]:
    snip_times: list[float] = []
    commitment_times: list[float] = []
    total_times: list[float] = []
    cfg = protocol_config(profile)
    protocol = VMaskProtocol(cfg, dimension=dim, max_clients=N_MAX)
    _, _, _, q_update = build_snip_instance(profile, dim, 20260706 + 5 * dim + profile.r0)

    for rep in range(repeats + warmup_runs):
        gc.collect()
        start_total = time.perf_counter()
        x = protocol._to_int_list(q_update)
        for value in x:
            if value < -cfg.amax or value > cfg.amax:
                raise ValueError("quantized update coordinate is outside [-Amax,Amax]")

        rng = random.Random(20260706 + 20_000 + rep)
        limbs = [
            [rng.randint(-cfg.r0, cfg.r0) for _ in range(dim)]
            for _ in range(cfg.limbs)
        ]
        mask = protocol._compose_mask(limbs)
        y_source = [x[r] + mask[r] for r in range(dim)]
        delta = [1 if value < 0 else 0 for value in y_source]
        y = [value + delta[r] * cfg.modulus for r, value in enumerate(y_source)]
        shifted = [[limbs[ell][r] + cfg.r0 for r in range(dim)] for ell in range(cfg.limbs)]
        z = protocol._compose_mask(limbs)
        statement = SnipStatement(
            theta_digest=protocol.theta_digest,
            sid=f"bench-paired-{profile.name}-{dim}-{rep}",
            committee_ids=protocol.verification_ids,
            client_id=0,
            y=y,
        )
        witness = SnipWitness(x=x, limbs=limbs, shifted=shifted, delta=delta, z=z)

        start_proof = time.perf_counter()
        proof = protocol.snip.prove(statement, witness, seed=20260706 + 30_000 + rep)
        snip_elapsed = time.perf_counter() - start_proof
        if not proof.constraints.satisfied:
            raise AssertionError("valid client submission failed during cost benchmark")

        gamma = ProtectedSubmission(sid=statement.sid, client_id=0, y=y)
        materials = [
            (m.z_shares, m.sigma_share)
            for m in proof.committee_material
        ]
        start_commitment = time.perf_counter()
        commitment_bundle = generate_packet_commitments(
            statement,
            materials,
            cfg,
        )
        commitment_elapsed = time.perf_counter() - start_commitment
        committee_material = [
            CommitteeMaterial(
                sigma_share=m.sigma_share,
                z_shares=m.z_shares,
                salt=commitment_bundle.salts[h],
            )
            for h, m in enumerate(proof.committee_material)
        ]
        evidence = VerificationEvidence(
            relation_digest=proof.relation_digest,
            statement_digest=commitment_bundle.statement_digest,
            packet_commitments=commitment_bundle.commitments,
            committee_material=committee_material,
        )
        _ = (gamma, evidence, z)
        total_elapsed = time.perf_counter() - start_total
        if rep >= warmup_runs:
            snip_times.append(snip_elapsed)
            commitment_times.append(commitment_elapsed)
            total_times.append(total_elapsed)

    return {
        "snip_proving": snip_times,
        "commitment_generation": commitment_times,
        "client_total": total_times,
    }


def bench_commitment_verification(
    profile: Profile,
    dim: int,
    repeats: int,
    warmup_runs: int = 0,
) -> dict[str, list[float]]:
    cfg = protocol_config(profile)
    protocol = VMaskProtocol(cfg, dimension=dim, max_clients=N_MAX)
    _, _, _, q_update = build_snip_instance(
        profile,
        dim,
        20260706 + 7 * dim + profile.r0,
    )
    submission = protocol.make_submission(
        sid=f"bench-commitment-{profile.name}-{dim}",
        client_id=0,
        q_update=q_update,
        seed=20260706 + 40_000,
    )
    statement = SnipStatement(
        theta_digest=protocol.theta_digest,
        sid=submission.gamma.sid,
        committee_ids=protocol.verification_ids,
        client_id=submission.gamma.client_id,
        y=submission.gamma.y,
    )
    all_times: list[float] = []
    per_packet_times: list[float] = []
    for rep in range(repeats + warmup_runs):
        gc.collect()
        start = time.perf_counter()
        valid = [
            verify_packet_commitment(
                statement=statement,
                committee_position=h + 1,
                z_shares=material.z_shares,
                verification_share=material.sigma_share,
                salt=material.salt,
                commitment=submission.evidence.packet_commitments[h],
                cfg=cfg,
            )
            for h, material in enumerate(submission.evidence.committee_material)
        ]
        elapsed = time.perf_counter() - start
        if not all(valid):
            raise AssertionError("valid packet commitment failed during cost benchmark")
        if rep >= warmup_runs:
            all_times.append(elapsed)
            per_packet_times.append(elapsed / cfg.committee_size)
    return {
        "commitment_verification_all": all_times,
        "commitment_verification_per_packet": per_packet_times,
    }


def qfield_estimates(profile: Profile, dim: int, unit_times: list[float]) -> dict[str, list[float]]:
    w = workload(profile, dim)
    gx = w["Gx"]
    nout = w["Nout"]
    prove_units = gx
    verify_units = gx
    recon_units = THRESHOLD * nout
    return {
        "prove": [u * prove_units for u in unit_times],
        "verify": [u * verify_units for u in unit_times],
        "reconstruct": [u * recon_units for u in unit_times],
    }


def communication(profile: Profile, dim: int) -> dict[str, float | int]:
    cfg = protocol_config(profile)
    counts = component_byte_counts(cfg, dimension=dim)
    return {
        "masked_update_bytes": counts.masked_update_bytes,
        "certified_recovery_share_bytes": counts.certified_recovery_share_bytes,
        "verification_material_bytes": counts.verification_material_bytes,
        "commitment_control_bytes": counts.commitment_control_bytes,
        "commitment_bytes": counts.commitment_bytes,
        "salt_bytes": counts.salt_bytes,
        "metadata_bytes": counts.metadata_bytes,
        "vmask_specific_client_payload_bytes": counts.vmask_specific_client_payload_bytes,
        "implemented_payload_subtotal_bytes": counts.implemented_payload_subtotal_bytes,
        "challenge_seed_bytes": counts.challenge_seed_bytes,
        "residual_reconstruction_bytes": counts.residual_reconstruction_bytes,
        "aggregate_recovery_communication_bytes": counts.aggregate_recovery_communication_bytes,
    }


def mb(value: float | int) -> float:
    return float(value) / 1_000_000.0


def computation_table_rows(time_rows: list[dict]) -> str:
    indexed = {
        (row["model"], row["profile"]): row
        for row in time_rows
    }

    def value(model: str, profile: str, metric: str) -> str:
        row = indexed[(model, profile)]
        median = float(row[f"{metric}_median_s"])
        iqr = float(row[f"{metric}_iqr_s"])
        return f"${median:.3f}\\,({iqr:.3f})$"

    components = [
        (r"\multirow{4}{*}{Client}", "Relation processing", "snip_proving", False),
        ("", "Commitment generation", "commitment_generation", False),
        ("", "Masking and submission", "masking_and_submission", False),
        ("", r"\textbf{Workload subtotal}", "client_total", True),
        (r"\multirow{4}{*}{Committee}", "Calibrated arithmetic", "committee_verification_workload", False),
        ("", "Commitment verification", "commitment_verification_all", False),
        ("", "Aggregate recovery", "aggregate_recovery", False),
        ("", r"\textbf{Workload subtotal}", "committee_total", True),
    ]
    lines = [
        r"\begin{table}[t]",
        r"	\centering",
        r"	\caption{Computation time of VMASK components (s).}",
        r"	\label{tab:cost-time}",
        r"	\footnotesize",
        r"	\setlength{\tabcolsep}{2.5pt}",
        r"	\renewcommand{\arraystretch}{1.10}",
        r"	\begin{tabularx}{\columnwidth}{@{}llXcc@{}}",
        r"		\toprule",
        r"		\textbf{Dataset} & \textbf{Side} & \textbf{Component} &",
        r"		\textbf{P80} & \textbf{P100} \\",
        r"		\midrule",
    ]
    for model_index, (model, _) in enumerate(MODELS):
        for component_index, (side, label, metric, bold) in enumerate(components):
            dataset_cell = rf"\multirow{{8}}{{*}}{{{model}}}" if component_index == 0 else ""
            side_cell = side
            p80 = value(model, "P80", metric)
            p100 = value(model, "P100", metric)
            if bold:
                p80 = rf"\mathbf{{{p80[1:-1]}}}$" if p80.startswith("$") else p80
                p100 = rf"\mathbf{{{p100[1:-1]}}}$" if p100.startswith("$") else p100
                p80 = "$" + p80
                p100 = "$" + p100
            lines.append(
                rf"		{dataset_cell} & {side_cell} & {label} & {p80} & {p100} \\"
            )
            if component_index == 3:
                lines.append(r"		\cmidrule(lr){2-5}")
        if model_index + 1 < len(MODELS):
            lines.append(r"		\midrule")
    lines.extend(
        [
            r"		\bottomrule",
            r"	\end{tabularx}",
            r"	\vspace{2pt}",
            r"	\begin{minipage}{0.98\columnwidth}",
            r"		\footnotesize",
            r"		Values are reported as median (IQR) over $10$ runs. Client workload",
            r"		subtotals are measured end to end. Committee workload subtotals combine",
            r"		calibrated arithmetic costs with measured commitment verification and recovery.",
            r"	\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def communication_table_rows(comm_rows: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"	\centering",
        r"	\caption{Serialized communication components of VMASK.}",
        r"	\label{tab:communication-cost}",
        r"	\footnotesize",
        r"	\setlength{\tabcolsep}{3.6pt}",
        r"	\renewcommand{\arraystretch}{1.08}",
        r"	\begin{tabular}{@{}llrrrrrr@{}}",
        r"		\toprule",
        r"		\textbf{Profile} & \textbf{Model} & \textbf{Masked (MB)} &",
        r"		\textbf{Recovery shares (MB)} & \textbf{Commit./control (kB)} &",
        r"		\textbf{Metadata (kB)} & \textbf{Impl. subtotal (MB)} &",
        r"		\textbf{Aggregate recovery (MB)} \\",
        r"		\midrule",
    ]
    for row in comm_rows:
        lines.extend(
            [
                rf"		{row['profile']}",
                r"		&",
                rf"		{row['model']}",
                r"		&",
                rf"		${mb(row['masked_update_bytes']):.3f}$",
                r"		&",
                rf"		${mb(row['certified_recovery_share_bytes']):.3f}$",
                r"		&",
                rf"		${row['commitment_control_bytes'] / 1_000:.3f}$",
                r"		&",
                rf"		${row['metadata_bytes'] / 1_000:.3f}$",
                r"		&",
                rf"		${mb(row['implemented_payload_subtotal_bytes']):.3f}$",
                r"		&",
                rf"		${mb(row['aggregate_recovery_communication_bytes']):.3f}$",
                r"		\\",
            ]
        )
    lines.extend(
        [
            r"		\bottomrule",
            r"	\end{tabular}",
            r"	\vspace{2pt}",
            r"",
            r"	\begin{minipage}{\linewidth}",
            r"		\footnotesize",
            r"		One MB and one kB denote $10^6$ and $10^3$ bytes, respectively.",
            r"		For $n=5$, the public commitment vector and the five packet salts",
            r"		each occupy $160$ bytes. Metadata contains a $32$-byte session",
            r"		digest, an $8$-byte client identifier, and the commitment vector.",
            r"		The term $p_{\mathsf{snip}}$ denotes client-to-committee communication",
            r"		required by a threshold-SNIP instantiation, excluding the recovery shares,",
            r"		packet salts, and public commitment metadata reported separately.",
            r"	\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_communication_outputs(out_dir: Path) -> list[dict]:
    comm_rows: list[dict] = []
    for model, dim in MODELS:
        for profile in PROFILES:
            comm = communication(profile, dim)
            comm_rows.append({"profile": profile.name, "model": model, "d": dim, **comm})
    write_csv(out_dir / "communication_summary.csv", comm_rows)
    (out_dir / "communication_table.tex").write_text(
        communication_table_rows(comm_rows),
        encoding="utf-8",
    )
    return comm_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def raw_rows(
    profile: str,
    model: str,
    dim: int,
    operation: str,
    values: list[float],
    source: str,
) -> list[dict]:
    return [
        {
            "profile": profile,
            "model": model,
            "d": dim,
            "operation": operation,
            "repeat": repeat,
            "seconds": value,
            "source": source,
        }
        for repeat, value in enumerate(values)
    ]


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": "VMASK protocol cost",
        "models": [{"name": model, "d": dim} for model, dim in MODELS],
        "profiles": [
            {
                "name": profile.name,
                "R0": profile.r0,
                "L": profile.limbs,
                "beta": profile.beta,
                "A_max": profile.amax,
            }
            for profile in PROFILES
        ],
        "N_max": N_MAX,
        "committee_n": COMMITTEE_N,
        "threshold": THRESHOLD,
        "field_q": str(Q_SNIP),
        "field_bytes": FIELD_BYTES,
        "aggregation_element_bytes": MASKED_UPDATE_BYTES,
        "timing_repeats": args.repeats,
        "snip_repeats": args.snip_repeats,
        "warmup_runs": args.warmup_runs,
        "affinity_cpu": args.affinity_cpu,
        "field_samples_per_repeat": args.field_samples,
        "statistic": "median and interquartile range",
        "python_version": sys.version,
        # Record a portable executable name rather than a local absolute path.
        "python_executable": Path(sys.executable).stem,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "benchmark_point_order": [
            {"profile": profile.name, "model": model, "d": dim}
            for model, dim in MODELS
            for profile in PROFILES
        ],
        "communication_mode": "fixed-width serialized protocol components",
        "communication_fields": [
            "masked_update_bytes",
            "certified_recovery_share_bytes",
            "verification_material_bytes",
            "commitment_control_bytes",
            "commitment_bytes",
            "salt_bytes",
            "metadata_bytes",
            "vmask_specific_client_payload_bytes",
            "implemented_payload_subtotal_bytes",
            "challenge_seed_bytes",
            "residual_reconstruction_bytes",
            "aggregate_recovery_communication_bytes",
        ],
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def make_figure(path: Path, rows: list[dict]) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    operations = [
        ("client_total", "Client workload subtotal"),
        ("committee_verification_workload", "Calibrated arithmetic workload"),
        ("aggregate_recovery", "Aggregate recovery"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    colors = {"P80": "#4C78A8", "P100": "#F58518"}
    for ax, (prefix, title) in zip(axes, operations):
        for profile in ["P80", "P100"]:
            subset = [r for r in rows if r["profile"] == profile]
            subset = sorted(subset, key=lambda r: int(r["d"]))
            x = [int(r["d"]) for r in subset]
            y = [float(r[f"{prefix}_median_s"]) for r in subset]
            err = [float(r[f"{prefix}_iqr_s"]) / 2.0 for r in subset]
            ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.8, capsize=3, label=profile, color=colors[profile])
        ax.set_title(title)
        ax.set_xlabel("Verified dimension")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis="x", style="plain")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VMASK protocol cost workloads")
    parser.add_argument("--out-dir", default="results/protocol_cost_salted_commitment_v1")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--snip-repeats", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--affinity-cpu", type=int)
    parser.add_argument("--field-samples", type=int, default=1_000_000)
    parser.add_argument("--communication-only", action="store_true")
    parser.add_argument("--skip-scalability", action="store_true")
    args = parser.parse_args()

    pin_current_process(args.affinity_cpu)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.communication_only:
        write_communication_outputs(out_dir)
        print(f"Wrote communication results to {out_dir}")
        return 0

    unit_times = calibration_samples(args.repeats, args.field_samples, args.warmup_runs)
    unit_median, unit_iqr = median_iqr(unit_times)

    workload_rows: list[dict] = []
    time_rows: list[dict] = []
    raw_time_rows: list[dict] = []
    raw_field_rows: list[dict] = [
        {
            "operation": "field_arithmetic_unit",
            "repeat": repeat,
            "seconds_per_operation": value,
            "field_samples": args.field_samples,
        }
        for repeat, value in enumerate(unit_times)
    ]
    comm_rows: list[dict] = []
    scalability_rows: list[dict] = []
    raw_scalability_rows: list[dict] = []

    benchmark_points = [(profile, model, dim) for model, dim in MODELS for profile in PROFILES]
    for profile, model, dim in benchmark_points:
        print(f"Benchmarking {profile.name} {model} (d={dim})", flush=True)
        w = workload(profile, dim)
        workload_rows.append(
            {
                "profile": profile.name,
                "model": model,
                "d": dim,
                "Gx": w["Gx"],
                "Nout": w["Nout"],
            }
        )
        client_breakdown = bench_client_submission_breakdown(
            profile,
            dim,
            args.snip_repeats,
            args.warmup_runs,
        )
        snip_proving = client_breakdown["snip_proving"]
        commitment_generation = client_breakdown["commitment_generation"]
        client_total = client_breakdown["client_total"]
        client_other = [
            total - snip - commitment
            for total, snip, commitment in zip(
                client_total,
                snip_proving,
                commitment_generation,
            )
        ]
        commitment_verification = bench_commitment_verification(
            profile,
            dim,
            args.repeats,
            args.warmup_runs,
        )
        commitment_verify_all = commitment_verification[
            "commitment_verification_all"
        ]
        commitment_verify_per_packet = commitment_verification[
            "commitment_verification_per_packet"
        ]
        recovery_base = bench_aggregate_recovery(
            profile,
            dim,
            args.repeats,
            args.warmup_runs,
        )
        q_est = qfield_estimates(profile, dim, unit_times)
        verification = q_est["verify"]
        verification_with_commitment = [
            arithmetic + packet
            for arithmetic, packet in zip(verification, commitment_verify_all)
        ]
        verification_median, _ = median_iqr(verification_with_commitment)
        verification_20 = N_MAX * verification_median
        aggregate_recovery = [a + b for a, b in zip(recovery_base, q_est["reconstruct"])]
        committee_total = [
            arithmetic + packet + recovery
            for arithmetic, packet, recovery in zip(
                verification,
                commitment_verify_all,
                aggregate_recovery,
            )
        ]
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "masking_and_submission",
                client_other,
                "paired_client_total_minus_snip",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "commitment_generation",
                commitment_generation,
                "salted_sha256_packet_commitments",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "snip_proving",
                snip_proving,
                "bench_client_submission_breakdown",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "commitment_verification_all",
                commitment_verify_all,
                "verify_all_packet_commitments",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "commitment_verification_per_packet",
                commitment_verify_per_packet,
                "verify_one_committee_packet",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "committee_total",
                committee_total,
                "verification_plus_aggregate_recovery",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "client_submission",
                client_total,
                "bench_client_submission_breakdown",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "committee_verification_workload",
                verification,
                "field_arithmetic_calibration",
            )
        )
        raw_time_rows.extend(
            raw_rows(
                profile.name,
                model,
                dim,
                "aggregate_recovery",
                aggregate_recovery,
                "bench_aggregate_recovery_plus_field_reconstruction",
            )
        )

        row = {
            "profile": profile.name,
            "model": model,
            "d": dim,
            "field_unit_median_s": unit_median,
            "field_unit_iqr_s": unit_iqr,
        }
        for name, values in [
            ("snip_proving", snip_proving),
            ("commitment_generation", commitment_generation),
            ("masking_and_submission", client_other),
            ("client_total", client_total),
            ("committee_verification_workload", verification),
            ("commitment_verification_all", commitment_verify_all),
            ("commitment_verification_per_packet", commitment_verify_per_packet),
            ("aggregate_recovery", aggregate_recovery),
            ("committee_total", committee_total),
        ]:
            med, iqr = median_iqr(values)
            row[f"{name}_median_s"] = med
            row[f"{name}_iqr_s"] = iqr
        row["estimated_sequential_verification_20_s"] = verification_20
        time_rows.append(row)

        comm = communication(profile, dim)
        comm_rows.append({"profile": profile.name, "model": model, "d": dim, **comm})

    if not args.skip_scalability:
        for profile in PROFILES:
            for dim in SCALABILITY_DIMS:
                client_breakdown = bench_client_submission_breakdown(
                    profile,
                    dim,
                    args.snip_repeats,
                    args.warmup_runs,
                )
                snip_proving = client_breakdown["snip_proving"]
                commitment_generation = client_breakdown["commitment_generation"]
                client_total = client_breakdown["client_total"]
                commitment_verification = bench_commitment_verification(
                    profile,
                    dim,
                    args.repeats,
                    args.warmup_runs,
                )
                commitment_verify_all = commitment_verification[
                    "commitment_verification_all"
                ]
                recovery_base = bench_aggregate_recovery(
                    profile,
                    dim,
                    args.repeats,
                    args.warmup_runs,
                )
                q_est = qfield_estimates(profile, dim, unit_times)
                verification = q_est["verify"]
                aggregate_recovery = [a + b for a, b in zip(recovery_base, q_est["reconstruct"])]
                for operation, values, source in [
                    ("snip_proving", snip_proving, "bench_client_submission_breakdown"),
                    ("commitment_generation", commitment_generation, "salted_sha256_packet_commitments"),
                    ("client_submission", client_total, "bench_client_submission_breakdown"),
                    ("committee_verification_workload", verification, "field_arithmetic_calibration"),
                    ("commitment_verification_all", commitment_verify_all, "verify_all_packet_commitments"),
                    (
                        "aggregate_recovery",
                        aggregate_recovery,
                        "bench_aggregate_recovery_plus_field_reconstruction",
                    ),
                ]:
                    for repeat, value in enumerate(values):
                        raw_scalability_rows.append(
                            {
                                "profile": profile.name,
                                "d": dim,
                                "operation": operation,
                                "repeat": repeat,
                                "seconds": value,
                                "source": source,
                            }
                        )
                row = {"profile": profile.name, "d": dim}
                for name, values in [
                    ("snip_proving", snip_proving),
                    ("commitment_generation", commitment_generation),
                    ("client_total", client_total),
                    ("committee_verification_workload", verification),
                    ("commitment_verification_all", commitment_verify_all),
                    ("aggregate_recovery", aggregate_recovery),
                ]:
                    med, iqr = median_iqr(values)
                    row[f"{name}_median_s"] = med
                    row[f"{name}_iqr_s"] = iqr
                comm = communication(profile, dim)
                row["aggregate_recovery_communication_mb"] = mb(
                    comm["aggregate_recovery_communication_bytes"]
                )
                scalability_rows.append(row)

    write_csv(out_dir / "workload_summary.csv", workload_rows)
    write_csv(out_dir / "time_summary.csv", time_rows)
    write_csv(out_dir / "raw_time_runs.csv", raw_time_rows)
    write_csv(out_dir / "raw_field_calibration.csv", raw_field_rows)
    write_csv(out_dir / "communication_summary.csv", comm_rows)
    (out_dir / "communication_table.tex").write_text(
        communication_table_rows(comm_rows),
        encoding="utf-8",
    )
    (out_dir / "computation_table.tex").write_text(
        computation_table_rows(time_rows),
        encoding="utf-8",
    )
    if scalability_rows:
        write_csv(out_dir / "dimension_scalability_summary.csv", scalability_rows)
        write_csv(out_dir / "raw_dimension_scalability_runs.csv", raw_scalability_rows)
        make_figure(out_dir / "fig6_dimension_scalability.png", scalability_rows)
    write_config(out_dir / "config.json", args)

    with (out_dir / "paper_tables.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"field_unit_median_s={unit_median:.9e}\n")
        handle.write(f"field_unit_iqr_s={unit_iqr:.9e}\n\n")
        handle.write("Computation table rows:\n")
        for row in time_rows:
            handle.write(
                f"{row['profile']} {row['model']}: "
                f"snip_proving {fmt_sec(row['snip_proving_median_s'])} ({fmt_sec(row['snip_proving_iqr_s'])}), "
                f"commitment_generation {fmt_sec(row['commitment_generation_median_s'])} ({fmt_sec(row['commitment_generation_iqr_s'])}), "
                f"masking_submission {fmt_sec(row['masking_and_submission_median_s'])} ({fmt_sec(row['masking_and_submission_iqr_s'])}), "
                f"client_total {fmt_sec(row['client_total_median_s'])} ({fmt_sec(row['client_total_iqr_s'])}), "
                f"verify_workload {fmt_sec(row['committee_verification_workload_median_s'])} ({fmt_sec(row['committee_verification_workload_iqr_s'])}), "
                f"commitment_verify_all {fmt_sec(row['commitment_verification_all_median_s'])} ({fmt_sec(row['commitment_verification_all_iqr_s'])}), "
                f"commitment_verify_packet {fmt_sec(row['commitment_verification_per_packet_median_s'])} ({fmt_sec(row['commitment_verification_per_packet_iqr_s'])}), "
                f"verify20_est {fmt_sec(row['estimated_sequential_verification_20_s'])}, "
                f"recover {fmt_sec(row['aggregate_recovery_median_s'])} ({fmt_sec(row['aggregate_recovery_iqr_s'])}), "
                f"committee_total {fmt_sec(row['committee_total_median_s'])} ({fmt_sec(row['committee_total_iqr_s'])})\n"
            )
        handle.write("\nCommunication table rows:\n")
        for row in comm_rows:
            handle.write(
                f"{row['profile']} {row['model']}: "
                f"masked {mb(row['masked_update_bytes']):.3f} MB, "
                f"certified_recovery {mb(row['certified_recovery_share_bytes']):.3f} MB, "
                f"commitment_control {row['commitment_control_bytes']} B, "
                f"D {row['commitment_bytes']} B, "
                f"salts {row['salt_bytes']} B, "
                f"metadata {row['metadata_bytes']} B, "
                f"vmask_specific_payload {mb(row['vmask_specific_client_payload_bytes']):.3f} MB, "
                f"implemented_subtotal {mb(row['implemented_payload_subtotal_bytes']):.6f} MB, "
                f"challenge {row['challenge_seed_bytes']} B, "
                f"residual_reconstruction {row['residual_reconstruction_bytes']} B, "
                f"aggregate_recovery {mb(row['aggregate_recovery_communication_bytes']):.3f} MB\n"
            )
        if scalability_rows:
            handle.write("\nDimension communication rows:\n")
            for dim in SCALABILITY_DIMS:
                p80 = next(r for r in scalability_rows if r["profile"] == "P80" and r["d"] == dim)
                p100 = next(r for r in scalability_rows if r["profile"] == "P100" and r["d"] == dim)
                handle.write(
                    f"{dim}: P80 {fmt_mib(p80['aggregate_recovery_communication_mb'])} MB, "
                    f"P100 {fmt_mib(p100['aggregate_recovery_communication_mb'])} MB\n"
                )

    print(f"Wrote benchmark results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
