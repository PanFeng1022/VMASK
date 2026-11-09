from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    ("mnist", "iid", ROOT / "results" / "amm_mnist_arithmetic_mismatch"),
    (
        "mnist",
        "dirichlet",
        ROOT / "results" / "amm_mnist_noniid_dirichlet_alpha0p5",
    ),
    (
        "cifar10",
        "iid",
        ROOT / "results" / "amm_cifar10_arithmetic_mismatch_500",
    ),
    (
        "cifar10",
        "dirichlet",
        ROOT / "results" / "amm_cifar10_noniid_dirichlet_alpha0p5_500",
    ),
]


def parse_source(raw: str) -> tuple[str, str, Path]:
    parts = raw.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "source must use DATASET|PARTITION|RESULTS_DIR syntax"
        )
    dataset, partition, path = parts
    if partition not in {"iid", "dirichlet"}:
        raise argparse.ArgumentTypeError("partition must be iid or dirichlet")
    return dataset, partition, Path(path)


def load_source(dataset: str, partition: str, results_dir: Path) -> pd.DataFrame:
    metric_files = sorted(results_dir.glob("gamma_*/*/seed_*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No AMM metrics found under {results_dir}")

    frames = []
    for path in metric_files:
        frame = pd.read_csv(path)
        frame["_metrics_path"] = str(path)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["dataset"] = dataset
    metrics["partition"] = partition
    metrics["alpha"] = 0.5 if partition == "dirichlet" else float("nan")
    return metrics


def select_complete_runs(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    keys = ["dataset", "partition", "gamma", "seed", "mode"]
    expected_rounds = {"mnist": 300, "cifar10": 500}
    candidates = []
    for path, group in metrics.groupby("_metrics_path", sort=True):
        rounds = group["round"].astype(int)
        first = group.iloc[0]
        config_path = Path(path).with_name("config.json")
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        candidates.append({
            "dataset": first["dataset"],
            "partition": first["partition"],
            "gamma": float(first["gamma"]),
            "seed": int(first["seed"]),
            "mode": first["mode"],
            "metrics_path": path,
            "row_count": int(len(group)),
            "min_round": int(rounds.min()),
            "max_round": int(rounds.max()),
            "unique_rounds": int(rounds.nunique()),
            "duplicate_round_rows": int(rounds.duplicated().sum()),
            "vmask_backend": config.get("vmask_backend", "full"),
            "protocol_execution": bool(config.get("protocol_execution", False)),
            "protocol_profile": config.get("protocol_profile"),
            "protocol_committee_size": config.get("protocol_committee_size"),
            "protocol_threshold": config.get("protocol_threshold"),
        })
    audit = pd.DataFrame(candidates)
    audit["expected_rounds"] = audit["dataset"].map(expected_rounds)
    audit["is_complete"] = (
        (audit["min_round"] == 1)
        & (audit["max_round"] == audit["expected_rounds"])
        & (audit["unique_rounds"] == audit["expected_rounds"])
        & (audit["duplicate_round_rows"] == 0)
    )
    audit["selected"] = False
    for _, group in audit.groupby(keys, sort=True):
        complete = group[group["is_complete"]]
        if len(complete) != 1:
            identity = group[keys].iloc[0].to_dict()
            raise RuntimeError(
                f"Expected exactly one complete raw result for {identity}; "
                f"found {len(complete)}"
            )
        audit.loc[complete.index[0], "selected"] = True
    selected_paths = set(audit.loc[audit["selected"], "metrics_path"])
    selected = metrics[metrics["_metrics_path"].isin(selected_paths)].copy()
    return selected, audit.sort_values(keys + ["selected", "metrics_path"])


def cumulative_error(group: pd.DataFrame) -> float:
    numerator_sq = float(group["agg_err_num_sq"].sum())
    denominator_sq = float(group["agg_err_den_sq"].sum())
    return math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + 1e-12)


def elapsed_seconds(group: pd.DataFrame) -> float:
    metrics_path = Path(str(group["_metrics_path"].iloc[0]))
    summary_path = metrics_path.with_name("run_summary.json")
    if not summary_path.exists():
        return float("nan")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(data["elapsed_seconds"])


def count_submissions(group: pd.DataFrame) -> tuple[int, int, int]:
    explicit_columns = {
        "malicious_submissions_constructed",
        "malicious_submissions_rejected",
        "malicious_submissions_accepted",
    }
    if explicit_columns.issubset(group.columns) and all(
        group[column].notna().all() for column in explicit_columns
    ):
        constructed = int(group["malicious_submissions_constructed"].sum())
        rejected = int(group["malicious_submissions_rejected"].sum())
        accepted = int(group["malicious_submissions_accepted"].sum())
        return constructed, rejected, accepted

    constructed = int(group["malicious_clients"].sum())
    rejected = int(
        round(float((group["malicious_clients"] * group["amm_reject_rate"]).sum()))
    )
    return constructed, rejected, constructed - rejected


def per_run_results(metrics: pd.DataFrame, final_window: int) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "partition", "alpha", "gamma", "seed", "mode"]
    for key, group in metrics.groupby(keys, dropna=False, sort=True):
        group = group.sort_values("round")
        max_round = int(group["round"].max())
        final = group[group["round"] > max_round - final_window]
        constructed, rejected, accepted = count_submissions(group)
        rows.append(
            {
                "dataset": key[0],
                "partition": key[1],
                "alpha": key[2],
                "gamma": float(key[3]),
                "seed": int(key[4]),
                "mode": key[5],
                "completed_rounds": max_round,
                "final_accuracy": float(final["test_accuracy"].mean()),
                "last_round_accuracy": float(group.iloc[-1]["test_accuracy"]),
                "AggErr_cum": cumulative_error(group),
                "malicious_submissions_constructed": constructed,
                "malicious_submissions_rejected": rejected,
                "malicious_submissions_accepted": accepted,
                "rejection_rate": rejected / constructed if constructed else 0.0,
                "elapsed_seconds": elapsed_seconds(group),
            }
        )
    return pd.DataFrame(rows)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "partition", "alpha", "gamma", "mode"]
    aggregate = (
        raw.groupby(keys, dropna=False, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_std=("final_accuracy", "std"),
            AggErr_cum_mean=("AggErr_cum", "mean"),
            AggErr_cum_std=("AggErr_cum", "std"),
            malicious_submissions_constructed=(
                "malicious_submissions_constructed",
                "sum",
            ),
            malicious_submissions_rejected=(
                "malicious_submissions_rejected",
                "sum",
            ),
            malicious_submissions_accepted=(
                "malicious_submissions_accepted",
                "sum",
            ),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
        )
        .fillna({
            "final_accuracy_std": 0.0,
            "AggErr_cum_std": 0.0,
            "elapsed_seconds_std": 0.0,
        })
    )
    aggregate["rejection_rate"] = (
        aggregate["malicious_submissions_rejected"]
        / aggregate["malicious_submissions_constructed"]
    )
    return aggregate


def paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "partition", "alpha", "gamma"]
    for index, group in aggregate.groupby(keys, dropna=False, sort=True):
        dataset, partition, alpha, gamma = index
        off = aggregate[
            (aggregate["dataset"] == dataset)
            & (aggregate["partition"] == partition)
            & (aggregate["gamma"] == gamma)
            & (aggregate["mode"] == "verification_off")
        ].iloc[0]
        protected = aggregate[
            (aggregate["dataset"] == dataset)
            & (aggregate["partition"] == partition)
            & (aggregate["gamma"] == gamma)
            & (aggregate["mode"] == "vmask")
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "partition": partition,
                "alpha": alpha,
                "gamma": gamma,
                "no_verification_accuracy": (
                    f"{100 * off['final_accuracy_mean']:.2f} +/- "
                    f"{100 * off['final_accuracy_std']:.2f}"
                ),
                "AggErr_cum": (
                    f"{off['AggErr_cum_mean']:.4f} +/- {off['AggErr_cum_std']:.4f}"
                ),
                "vmask_accuracy": (
                    f"{100 * protected['final_accuracy_mean']:.2f} +/- "
                    f"{100 * protected['final_accuracy_std']:.2f}"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["_dataset_order"] = table["dataset"].map({"mnist": 0, "cifar10": 1})
    table["_partition_order"] = table["partition"].map({"iid": 0, "dirichlet": 1})
    return (
        table.sort_values(
            ["_dataset_order", "_partition_order", "gamma"], kind="stable"
        )
        .drop(columns=["_dataset_order", "_partition_order"])
    )


def write_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Partition & $\gamma$ & No verification (\%) & $\mathrm{AggErr}_{\mathrm{cum}}$ & VMASK (\%) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        dataset = "MNIST" if row["dataset"] == "mnist" else "CIFAR-10"
        partition = (
            "IID"
            if row["partition"] == "iid"
            else r"Non-IID ($\alpha=0.5$)"
        )
        off = str(row["no_verification_accuracy"]).replace(" +/- ", r"\pm")
        aggerr = str(row["AggErr_cum"]).replace(" +/- ", r"\pm")
        protected = str(row["vmask_accuracy"]).replace(" +/- ", r"\pm")
        lines.append(
            rf"{dataset} & {partition} & {row['gamma']:.1f} & "
            rf"${off}$ & ${aggerr}$ & ${protected}$ "
            rf"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the IID/Non-IID AMM revision tables"
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help="Repeatable DATASET|PARTITION|RESULTS_DIR source",
    )
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "amm_noniid_revision"),
    )
    args = parser.parse_args()

    sources = args.source or DEFAULT_SOURCES
    loaded_metrics = pd.concat(
        [load_source(dataset, partition, path) for dataset, partition, path in sources],
        ignore_index=True,
    )
    metrics, audit = select_complete_runs(loaded_metrics)
    raw = per_run_results(metrics, args.final_window)
    aggregate = aggregate_results(raw)
    table = paper_table(aggregate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "source_file_audit.csv", index=False)
    raw.to_csv(output_dir / "raw_run_results.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    table.to_csv(output_dir / "table_vi_extended.csv", index=False)
    write_latex(table, output_dir / "table_vi_extended.tex")

    failures = raw[
        (raw["partition"] == "dirichlet")
        & (raw["mode"] == "vmask")
        & (raw["rejection_rate"] < 1.0 - 1e-12)
    ]
    failures.to_csv(output_dir / "rejection_failures.csv", index=False)

    noniid = raw[raw["partition"] == "dirichlet"]
    expected_keys = {
        (dataset, gamma, seed, mode)
        for dataset in ("mnist", "cifar10")
        for gamma in (0.1, 0.2, 0.3)
        for seed in (8207, 9090, 9421, 7515, 4407)
        for mode in ("verification_off", "vmask")
    }
    actual_keys = set(
        noniid[["dataset", "gamma", "seed", "mode"]]
        .itertuples(index=False, name=None)
    )
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    selected_noniid_audit = audit[
        (audit["partition"] == "dirichlet") & audit["selected"]
    ]
    selected_noniid_vmask = selected_noniid_audit[
        selected_noniid_audit["mode"] == "vmask"
    ]
    selected_vmask = audit[audit["selected"] & (audit["mode"] == "vmask")]
    invalid_vmask = selected_vmask[
        (~selected_vmask["protocol_execution"])
        | (selected_vmask["vmask_backend"] != "full")
    ]
    if not invalid_vmask.empty:
        raise RuntimeError(
            "Every selected VMASK run must use protocol_execution=true and "
            "vmask_backend=full"
        )
    full_protocol_vmask = selected_noniid_vmask[
        selected_noniid_vmask["protocol_execution"]
        & (selected_noniid_vmask["vmask_backend"] == "full")
    ]
    completeness = {
        "expected_jobs": 60,
        "selected_complete_jobs": int(len(noniid)),
        "missing_jobs": missing,
        "unexpected_jobs": unexpected,
        "full_protocol_vmask_jobs_expected": 30,
        "full_protocol_vmask_jobs_complete": int(len(full_protocol_vmask)),
        "all_recorded_vmask_amm_submissions_marked_rejected": bool(
            (noniid[noniid["mode"] == "vmask"]["malicious_submissions_accepted"] == 0).all()
            and (noniid[noniid["mode"] == "vmask"]["rejection_rate"] == 1.0).all()
        ),
        "all_vmask_jobs_executed_full_protocol": bool(len(full_protocol_vmask) == 30),
        "evidence_fields": [
            "malicious_submissions_constructed",
            "malicious_submissions_rejected",
            "malicious_submissions_accepted",
        ],
    }
    (output_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    lines = [
        "Non-IID AMM completeness verification",
        f"Expected jobs: {completeness['expected_jobs']}",
        f"Selected complete jobs: {completeness['selected_complete_jobs']}",
        f"Missing jobs: {len(missing)}",
        f"Unexpected jobs: {len(unexpected)}",
        f"Full-protocol VMASK jobs: {len(full_protocol_vmask)}/30",
        "All recorded VMASK AMM submissions marked rejected: "
        + str(completeness["all_recorded_vmask_amm_submissions_marked_rejected"]),
        "Evidence: sums of the three per-round malicious_submissions_* fields "
        "in complete native VMASK metrics.csv files.",
    ]
    (output_dir / "verification_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if missing or unexpected or len(noniid) != 60 or len(full_protocol_vmask) != 30:
        raise RuntimeError(f"Non-IID matrix validation failed: {completeness}")
    if not failures.empty:
        print("WARNING: Non-IID VMASK rejection validation failed")
        print(failures.to_string(index=False))

    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
