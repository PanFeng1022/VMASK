from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "external"
DEPS_PY39 = ROOT / ".deps" / "python39"
if DEPS_PY39.exists() and sys.version_info[:2] == (3, 9):
    sys.path.insert(0, str(DEPS_PY39))

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, NullLocator
except ModuleNotFoundError:
    plt = None
    LogFormatterMathtext = LogLocator = NullFormatter = NullLocator = None

import numpy as np

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_protocol_cost import (
    Profile,
    bench_aggregate_recovery,
    bench_client_submission_breakdown,
    bench_commitment_verification,
    median_iqr,
    pin_current_process,
    qfield_estimates,
)
from vmask_exp.protocol import VMaskProtocol
from vmask_exp.protocol.config import profile_p100, profile_p80


N_CLIENTS = 20
COMMITTEE_N = 5
THRESHOLD = 3
DIMS = [1_000, 10_000, 27_562, 105_050]
SEED = 20260706
PIROAGG_F = 4
PIROAGG_K_LCC = 14
PROTOCOL_COST_SUMMARY = (
    ROOT / "results" / "protocol_cost_salted_commitment_v1" / "time_summary.csv"
)

VMASK_PROFILES = [
    Profile(name="P80", r0=1023, limbs=10, beta=2047),
    Profile(name="P100", r0=8191, limbs=9, beta=16383),
]


@dataclass
class RunRecord:
    scheme: str
    profile: str
    dimension: int
    side: str
    repeat: int
    seconds: float
    benchmark_scope: str


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def protocol_config(profile: Profile):
    if profile.name.endswith("P80"):
        return profile_p80(committee_size=COMMITTEE_N, threshold=THRESHOLD)
    if profile.name.endswith("P100"):
        return profile_p100(committee_size=COMMITTEE_N, threshold=THRESHOLD)
    raise ValueError(f"unsupported VMASK profile: {profile.name}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_protocol_cost_field_unit() -> float:
    if not PROTOCOL_COST_SUMMARY.exists():
        raise FileNotFoundError(
            f"protocol cost summary not found: {PROTOCOL_COST_SUMMARY}"
        )
    with PROTOCOL_COST_SUMMARY.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {
        float(row["field_unit_median_s"])
        for row in rows
        if row.get("field_unit_median_s")
    }
    if not values:
        raise ValueError(f"missing field_unit_median_s in {PROTOCOL_COST_SUMMARY}")
    if len(values) != 1:
        raise ValueError(
            f"inconsistent field_unit_median_s values in {PROTOCOL_COST_SUMMARY}: {sorted(values)}"
        )
    return values.pop()


def load_protocol_cost_field_units() -> list[float]:
    path = PROTOCOL_COST_SUMMARY.with_name("raw_field_calibration.csv")
    if not path.exists():
        return [load_protocol_cost_field_unit()]
    with path.open("r", newline="", encoding="utf-8") as handle:
        values = [
            float(row["seconds_per_operation"])
            for row in csv.DictReader(handle)
        ]
    if not values:
        raise ValueError(f"missing field calibration samples in {path}")
    return values


def load_protocol_cost_raw_operations() -> dict[tuple[str, int, str], list[float]]:
    path = PROTOCOL_COST_SUMMARY.with_name("raw_time_runs.csv")
    if not path.exists():
        return {}
    grouped: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["profile"], int(row["d"]), row["operation"])
            grouped.setdefault(key, []).append(
                (int(row["repeat"]), float(row["seconds"]))
            )
    return {
        key: [value for _, value in sorted(values)]
        for key, values in grouped.items()
    }


def load_protocol_cost_snip_proving() -> dict[tuple[str, int], tuple[float, float]]:
    if not PROTOCOL_COST_SUMMARY.exists():
        raise FileNotFoundError(
            f"protocol cost summary not found: {PROTOCOL_COST_SUMMARY}"
        )
    values: dict[tuple[str, int], tuple[float, float]] = {}
    with PROTOCOL_COST_SUMMARY.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = row["profile"]
            dim = int(row["d"])
            values[(profile, dim)] = (
                float(row["snip_proving_median_s"]),
                float(row["snip_proving_iqr_s"]),
            )
    if not values:
        raise ValueError(f"missing SNIP proving data in {PROTOCOL_COST_SUMMARY}")
    return values


def align_vmask_client_proof_with_protocol_cost(summary_rows: list[dict]) -> None:
    shared = load_protocol_cost_snip_proving()
    for row in summary_rows:
        if row["scheme"] != "VMASK" or row["side"] != "client_workload":
            continue
        key = (row["profile"], int(row["dimension"]))
        if key not in shared:
            continue
        median, iqr = shared[key]
        row["median_seconds"] = f"{median:.6f}"
        row["iqr_seconds"] = f"{iqr:.6f}"
        row["benchmark_scope"] = (
            "relation processing shared with protocol-cost table"
        )


def time_vmask_client_processing(profile: Profile, dim: int, repeats: int, warmups: int) -> list[float]:
    return bench_client_submission_breakdown(
        profile,
        dim,
        repeats,
        warmups,
    )["client_total"]


def time_vmask_aggregation_verification(
    profile: Profile,
    dim: int,
    repeats: int,
    warmups: int,
    field_unit_times: list[float],
) -> list[float]:
    if len(field_unit_times) < repeats:
        raise ValueError("insufficient field calibration samples")
    unit_times = field_unit_times[:repeats]
    verification_times = qfield_estimates(profile, dim, unit_times)["verify"]
    commitment_times = bench_commitment_verification(
        profile,
        dim,
        repeats,
        warmups,
    )["commitment_verification_all"]
    reconstruction_times = qfield_estimates(profile, dim, unit_times)["reconstruct"]
    recovery_base = bench_aggregate_recovery(profile, dim, repeats, warmups)
    recovery_times = [
        base + reconstruct
        for base, reconstruct in zip(recovery_base, reconstruction_times)
    ]
    return [
        N_CLIENTS * (verification + commitment) + recovery
        for verification, commitment, recovery in zip(
            verification_times,
            commitment_times,
            recovery_times,
        )
    ]


def record_vmask(
    repeats: int,
    warmups: int,
    dims: Iterable[int],
    field_unit_times: list[float],
) -> list[RunRecord]:
    records: list[RunRecord] = []
    shared = load_protocol_cost_raw_operations()
    for profile in VMASK_PROFILES:
        for dim in dims:
            print(f"[VMASK] profile={profile.name} d={dim}: client workload", flush=True)
            prove_times = shared.get((profile.name, dim, "client_submission"))
            prove_scope = "relation processing, packet commitments, and client submission generation"
            if prove_times is None:
                prove_times = time_vmask_client_processing(profile, dim, repeats, warmups)
            else:
                prove_times = prove_times[:repeats]
                prove_scope = "client processing shared with salted-commitment protocol-cost run"
            print(f"[VMASK] profile={profile.name} d={dim}: aggregation-side verification workload", flush=True)
            shared_verify = shared.get(
                (profile.name, dim, "committee_verification_workload")
            )
            shared_commitment = shared.get(
                (profile.name, dim, "commitment_verification_all")
            )
            shared_recovery = shared.get((profile.name, dim, "aggregate_recovery"))
            verify_scope = (
                "aggregation-side relation workload for 20 submissions, using "
                "calibrated arithmetic costs plus measured commitment processing and recovery"
            )
            if (
                shared_verify is not None
                and shared_commitment is not None
                and shared_recovery is not None
            ):
                verify_times = [
                    N_CLIENTS * (verification + commitment) + recovery
                    for verification, commitment, recovery in zip(
                        shared_verify[:repeats],
                        shared_commitment[:repeats],
                        shared_recovery[:repeats],
                    )
                ]
                verify_scope += "; shared with controlled protocol-cost run"
            else:
                verify_times = time_vmask_aggregation_verification(
                    profile,
                    dim,
                    repeats,
                    warmups,
                    field_unit_times,
                )
            for idx, value in enumerate(prove_times):
                records.append(
                    RunRecord(
                        scheme="VMASK",
                        profile=profile.name.replace("VMASK-", ""),
                        dimension=dim,
                        side="client_workload",
                        repeat=idx,
                        seconds=value,
                        benchmark_scope=prove_scope,
                    )
                )
            for idx, value in enumerate(verify_times):
                records.append(
                    RunRecord(
                        scheme="VMASK",
                        profile=profile.name.replace("VMASK-", ""),
                        dimension=dim,
                        side="aggregation_workload",
                        repeat=idx,
                        seconds=value,
                        benchmark_scope=verify_scope,
                    )
                )
    return records


def priroagg_available() -> tuple[bool, str]:
    try:
        importlib.import_module("galois")
    except ModuleNotFoundError as exc:
        return False, f"missing dependency: {exc.name}"
    pri_dir = EXTERNAL_ROOT / "PriRoAgg"
    if not pri_dir.exists():
        return False, "external/PriRoAgg is not present"
    return True, "available"


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def import_priroagg():
    pri_dir = EXTERNAL_ROOT / "PriRoAgg"
    sys.path.insert(0, str(pri_dir))
    snip_module = importlib.import_module("snip_module")
    # The public PriRoAgg prototype uses math.sqrt in initial() without
    # importing math. Inject it here instead of modifying the external source.
    snip_module.math = math
    return snip_module


def time_priroagg_workload(dim: int, repeats: int, warmups: int) -> tuple[list[float], list[float]]:
    snip_module = import_priroagg()
    pri_dir = EXTERNAL_ROOT / "PriRoAgg"
    # These parameters follow the comparison setting. The public prototype
    # internally overwrites K in the SNIP object; this is recorded in config.
    with pushd(pri_dir):
        p, q, r, g = snip_module.initial(2 * 10**8)
        lcc_para = [p, q, r, g, PIROAGG_F, N_CLIENTS, PIROAGG_K_LCC]
        alphas = list(range(1, 1 + N_CLIENTS))
        betas = list(range(1 + N_CLIENTS, N_CLIENTS + PIROAGG_K_LCC + PIROAGG_F + 1))
        chunks = math.ceil(dim / PIROAGG_K_LCC)
        client_times: list[float] = []
        aggregation_times: list[float] = []
        for rep in range(warmups + repeats):
            gc.collect()
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                snip = snip_module.Snip(10, lcc_para, alphas, betas)
                start = time.perf_counter()
                snip.prove()
                prove_time = time.perf_counter() - start
                start = time.perf_counter()
                verify_time = snip.verify()
                lhs_shares, _ = snip.beaver()
                _ = time.perf_counter() - start
                start = time.perf_counter()
                snip.server_check(lhs_shares)
                server_check_time = time.perf_counter() - start
                # Follow the public prototype's overhead accounting in
                # LCC_module.py: one user proves once and verifies the other
                # N-1 users, while the server checks one SNIP instance for
                # each user. Both costs are scaled by the number of chunks.
                user_time = (prove_time + (N_CLIENTS - 1) * verify_time) * chunks
                server_time = server_check_time * N_CLIENTS * chunks
            if rep >= warmups:
                client_times.append(user_time)
                aggregation_times.append(server_time)
        return client_times, aggregation_times


def record_priroagg(repeats: int, warmups: int, dims: Iterable[int]) -> list[RunRecord]:
    records: list[RunRecord] = []
    ok, reason = priroagg_available()
    if not ok:
        print(f"[skip] PriRoAgg workload: {reason}", file=sys.stderr)
        return records
    for dim in dims:
        print(f"[PriRoAgg] d={dim}: public implementation workload", flush=True)
        client_times, aggregation_times = time_priroagg_workload(dim, repeats, warmups)
        for idx, value in enumerate(client_times):
            records.append(
                RunRecord(
                    scheme="PriRoAgg",
                    profile="workload",
                    dimension=dim,
                    side="client_workload",
                    repeat=idx,
                    seconds=value,
                    benchmark_scope="public implementation user SNIP workload",
                )
            )
        for idx, value in enumerate(aggregation_times):
            records.append(
                RunRecord(
                    scheme="PriRoAgg",
                    profile="workload",
                    dimension=dim,
                    side="aggregation_workload",
                    repeat=idx,
                    seconds=value,
                    benchmark_scope="public implementation server_check workload",
                )
            )
    return records


def load_external_records(path: Path | None) -> list[RunRecord]:
    if path is None:
        return []
    rows: list[RunRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                RunRecord(
                    scheme=row["scheme"],
                    profile=row.get("profile", "workload"),
                    dimension=int(row["dimension"]),
                    side=row["side"],
                    repeat=int(row.get("repeat", 0)),
                    seconds=float(row["seconds"]),
                    benchmark_scope=row.get("benchmark_scope", "external benchmark"),
                )
            )
    return rows


def summarize(records: list[RunRecord], repeats: int) -> list[dict]:
    grouped: dict[tuple[str, str, int, str, str], list[float]] = {}
    for rec in records:
        key = (rec.scheme, rec.profile, rec.dimension, rec.side, rec.benchmark_scope)
        grouped.setdefault(key, []).append(rec.seconds)
    rows: list[dict] = []
    for (scheme, profile, dim, side, scope), values in sorted(grouped.items()):
        median, iqr = median_iqr(values)
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        rows.append(
            {
                "scheme": scheme,
                "profile": profile,
                "dimension": dim,
                "side": side,
                "median_seconds": f"{median:.6f}",
                "iqr_seconds": f"{iqr:.6f}",
                "q1_seconds": f"{q1:.6f}",
                "q3_seconds": f"{q3:.6f}",
                "repeats": len(values),
                "target_repeats": repeats,
                "benchmark_scope": scope,
            }
        )
    return rows


def make_latex_table(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{@{}lllrrr@{}}\n")
        handle.write("\\toprule\n")
        handle.write("Scheme & Profile & Side & $d$ & Median (s) & IQR (s)\\\\\n")
        handle.write("\\midrule\n")
        for row in summary_rows:
            side = (
                "Client computation"
                if row["side"] == "client_workload"
                else "Aggregation computation"
            )
            handle.write(
                f"{row['scheme']} & {row['profile']} & {side} & "
                f"{int(row['dimension']):,} & {float(row['median_seconds']):.3f} & "
                f"{float(row['iqr_seconds']):.3f}\\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def comparison_ratio_rows(summary_rows: list[dict]) -> list[dict]:
    medians = {
        (
            row["scheme"],
            row["profile"],
            int(row["dimension"]),
            row["side"],
        ): float(row["median_seconds"])
        for row in summary_rows
    }
    dimensions = sorted(
        {
            int(row["dimension"])
            for row in summary_rows
            if row["scheme"] == "PriRoAgg"
        }
    )
    rows: list[dict] = []
    for dim in dimensions:
        priroagg_client = medians.get(
            ("PriRoAgg", "workload", dim, "client_workload")
        )
        priroagg_aggregation = medians.get(
            ("PriRoAgg", "workload", dim, "aggregation_workload")
        )
        if priroagg_client is None or priroagg_aggregation is None:
            continue
        for profile in ("P80", "P100"):
            vmask_client = medians.get(
                ("VMASK", profile, dim, "client_workload")
            )
            vmask_aggregation = medians.get(
                ("VMASK", profile, dim, "aggregation_workload")
            )
            if vmask_client is None or vmask_aggregation is None:
                continue
            rows.append(
                {
                    "dimension": dim,
                    "profile": f"VMASK-{profile}",
                    "vmask_client_median_s": f"{vmask_client:.6f}",
                    "priroagg_client_median_s": f"{priroagg_client:.6f}",
                    "client_reduction_factor": f"{priroagg_client / vmask_client:.6f}",
                    "vmask_aggregation_median_s": f"{vmask_aggregation:.6f}",
                    "priroagg_aggregation_median_s": f"{priroagg_aggregation:.6f}",
                    "priroagg_aggregation_reduction_factor": (
                        f"{vmask_aggregation / priroagg_aggregation:.6f}"
                    ),
                }
            )
    return rows


def write_comparison_claim(path: Path, ratio_rows: list[dict]) -> None:
    if not ratio_rows:
        return
    client_factors = [
        float(row["client_reduction_factor"])
        for row in ratio_rows
    ]
    aggregation_factors = [
        float(row["priroagg_aggregation_reduction_factor"])
        for row in ratio_rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Across the evaluated dimensions, VMASK reduces client computation "
        f"by a factor of approximately ${min(client_factors):.1f}$--"
        f"${max(client_factors):.1f}$ relative to PriRoAgg. In contrast, "
        "the aggregation computation of PriRoAgg is lower than that of VMASK "
        f"by a factor of approximately ${min(aggregation_factors):.1f}$--"
        f"${max(aggregation_factors):.1f}$.\n",
        encoding="utf-8",
    )


def make_figure(path: Path, summary_rows: list[dict]) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    sides = [
        ("client_workload", "(a) Client computation"),
        ("aggregation_workload", "(b) Aggregation computation"),
    ]
    styles = {
        ("VMASK", "P80"): ("#4C78A8", "o", "VMASK-P80"),
        ("VMASK", "P100"): ("#F58518", "s", "VMASK-P100"),
        ("PriRoAgg", "workload"): ("#54A24B", "^", "PriRoAgg"),
        ("ACORN", "workload"): ("#B279A2", "D", "ACORN workload"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharex=True)
    for ax, (side, panel_label) in zip(axes, sides):
        plotted_x: list[int] = []
        for key, (color, marker, label) in styles.items():
            scheme, profile = key
            subset = [
                row for row in summary_rows
                if row["side"] == side and row["scheme"] == scheme and row["profile"] == profile
            ]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: int(r["dimension"]))
            x = [int(r["dimension"]) for r in subset]
            y = [float(r["median_seconds"]) for r in subset]
            if all(r.get("q1_seconds") and r.get("q3_seconds") for r in subset):
                err = [
                    [
                        float(r["median_seconds"]) - float(r["q1_seconds"])
                        for r in subset
                    ],
                    [
                        float(r["q3_seconds"]) - float(r["median_seconds"])
                        for r in subset
                    ],
                ]
            else:
                half_iqr = [float(r["iqr_seconds"]) / 2.0 for r in subset]
                err = [half_iqr, half_iqr]
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=color,
                marker=marker,
                markersize=5.8,
                markeredgewidth=0.8,
                linewidth=1.5,
                elinewidth=1.3,
                capsize=2.8,
                capthick=1.0,
                label=label,
                barsabove=True,
                zorder=3,
            )
            plotted_x.extend(x)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if plotted_x:
            tick_values = sorted(set(plotted_x))
            ax.set_xlim(800, 130_000)
            ax.set_xticks(tick_values)
            ax.set_xticklabels(
                [f"{value:,}" for value in tick_values]
            )
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.grid(False)
        ax.yaxis.grid(
            True,
            which="major",
            color="#d8d8d8",
            linewidth=0.65,
            linestyle="-",
            alpha=0.85,
            zorder=0,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#222222")
        ax.tick_params(axis="both", which="major", length=3.0, width=0.75)
        ax.tick_params(axis="both", which="minor", length=0)
        ax.set_xlabel(r"Update dimension $d$")
        ax.text(0.5, -0.28, panel_label, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    axes[0].set_ylabel("Computation time (s)")
    axes[0].legend(
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d0d0",
        handlelength=1.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.085, right=0.98, top=0.98, bottom=0.30, wspace=0.22)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_config(
    path: Path,
    args: argparse.Namespace,
    included: list[str],
    skipped: list[str],
    field_unit_median: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        protocol_cost_source = PROTOCOL_COST_SUMMARY.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        protocol_cost_source = PROTOCOL_COST_SUMMARY.name
    config = {
        "benchmark": "prior work scalability",
        "scope": "relation-layer and public-implementation workloads",
        "N": N_CLIENTS,
        "dimensions": args.dims,
        "repeats": args.repeats,
        "warmup_runs": args.warmups,
        "affinity_cpu": args.affinity_cpu,
        "statistic": "median and IQR",
        "accepted_submission_count": N_CLIENTS,
        "vmask_field_unit_source": protocol_cost_source,
        "vmask_field_unit_median_s": field_unit_median,
        "vmask_client_proof_model_dimensions_source": protocol_cost_source,
        "timed_functions": {
            "VMASK": [
                "bench_client_submission_breakdown(...)[client_total]",
                "20 calibrated C1-C6 arithmetic workloads and packet checks plus measured aggregate recovery",
            ],
            "PriRoAgg": [
                "Snip.prove",
                "Snip.verify",
                "Snip.server_check",
            ],
        },
        "included_protocol_stages": [
            "VMASK SNIP constraint evaluation",
            "VMASK certified output sharing material generation",
            "VMASK client processing from the same benchmark path as the protocol-cost table",
            "VMASK salted packet commitment generation and verification",
            "VMASK estimated C1-C6 verification workload for 20 submissions",
            "VMASK verification residual reconstruction",
            "VMASK aggregate recovery",
            "PriRoAgg public implementation SNIP proving workload when dependencies are available",
            "PriRoAgg public implementation user SNIP workload when dependencies are available",
            "PriRoAgg public implementation server_check workload when dependencies are available",
        ],
        "excluded_protocol_stages": [
            "local model training",
            "network transport",
            "full end-to-end PriRoAgg protocol latency",
        ],
        "VMASK_profiles": ["P80", "P100"],
        "PriRoAgg": {
            "f": PIROAGG_F,
            "K_LCC": PIROAGG_K_LCC,
            "repository": "external/PriRoAgg",
            "commit": git_commit(EXTERNAL_ROOT / "PriRoAgg"),
        },
        "included_schemes": included,
        "skipped_schemes": skipped,
        "random_seed": SEED,
        "python_version": sys.version,
        # Record a portable executable name rather than a local absolute path.
        "python_executable": Path(sys.executable).stem,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    global PROTOCOL_COST_SUMMARY

    parser = argparse.ArgumentParser(description="Benchmark prior-work scalability in an isolated result directory")
    parser.add_argument("--out-dir", default="results/prior_work_scalability_salted_commitment_v1")
    parser.add_argument("--dims", type=int, nargs="+", default=DIMS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--affinity-cpu", type=int)
    parser.add_argument("--protocol-cost-summary", type=Path, default=PROTOCOL_COST_SUMMARY)
    parser.add_argument(
        "--plot-from-summary",
        type=Path,
        help="Regenerate the comparison figure from an existing scalability_summary.csv without rerunning benchmarks.",
    )
    parser.add_argument("--skip-vmask", action="store_true")
    parser.add_argument("--include-priroagg", action="store_true")
    parser.add_argument(
        "--external-runs-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional raw run CSV files to merge into the comparison.",
    )
    parser.add_argument("--acorn-csv", type=Path, default=None, help="Optional ACORN workload CSV with the raw schema")
    args = parser.parse_args()

    PROTOCOL_COST_SUMMARY = args.protocol_cost_summary
    pin_current_process(args.affinity_cpu)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    for directory in [raw_dir, tables_dir, figures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.plot_from_summary is not None:
        with args.plot_from_summary.open("r", newline="", encoding="utf-8-sig") as handle:
            summary_rows = list(csv.DictReader(handle))
        if not summary_rows:
            raise ValueError(f"no summary rows found in {args.plot_from_summary}")
        make_figure(figures_dir / "prior_work_scalability.png", summary_rows)
        print(f"wrote figure to {figures_dir}")
        return 0

    records: list[RunRecord] = []
    included: list[str] = []
    skipped: list[str] = []
    field_unit_median = load_protocol_cost_field_unit()
    field_unit_times = load_protocol_cost_field_units()

    if args.skip_vmask:
        skipped.append("VMASK")
    else:
        included.append("VMASK")
        records.extend(record_vmask(args.repeats, args.warmups, args.dims, field_unit_times))

    if args.include_priroagg:
        ok, reason = priroagg_available()
        if ok:
            included.append("PriRoAgg")
            records.extend(record_priroagg(args.repeats, args.warmups, args.dims))
        else:
            skipped.append(f"PriRoAgg ({reason})")
            print(f"[skip] PriRoAgg: {reason}", file=sys.stderr)
    else:
        skipped.append("PriRoAgg (not requested)")

    external: list[RunRecord] = []
    for csv_path in args.external_runs_csv:
        external.extend(load_external_records(csv_path))
    external.extend(load_external_records(args.acorn_csv))
    if external:
        external_schemes = sorted({record.scheme for record in external})
        included.extend(external_schemes)
        skipped = [
            entry
            for entry in skipped
            if not any(entry == scheme or entry.startswith(f"{scheme} ") for scheme in external_schemes)
        ]
        records.extend(external)
    else:
        skipped.append("ACORN (no artifact/workload CSV provided)")

    raw_rows = [record.__dict__ for record in records]
    write_csv(raw_dir / "runs.csv", raw_rows)
    summary_rows = summarize(records, args.repeats)
    write_csv(tables_dir / "scalability_summary.csv", summary_rows)
    make_latex_table(tables_dir / "scalability_summary.tex", summary_rows)
    ratio_rows = comparison_ratio_rows(summary_rows)
    write_csv(tables_dir / "comparison_ratios.csv", ratio_rows)
    write_comparison_claim(tables_dir / "comparison_claim.tex", ratio_rows)
    if summary_rows:
        make_figure(figures_dir / "prior_work_scalability.png", summary_rows)
    write_config(out_dir / "config.json", args, included, skipped, field_unit_median)

    with (logs_dir / "benchmark.log").open("w", encoding="utf-8") as handle:
        handle.write("included=" + ", ".join(included) + "\n")
        handle.write("skipped=" + ", ".join(skipped) + "\n")
        handle.write(f"records={len(records)}\n")

    print(f"wrote results to {out_dir}")
    if skipped:
        print("skipped: " + "; ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
