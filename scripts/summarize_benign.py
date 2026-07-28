from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{100.0 * mean:.2f} +/- {100.0 * std:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benign MNIST runs")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "benign_mnist_300_seedcheck"))
    parser.add_argument("--final-window", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metric_files = sorted(results_dir.glob("seed_*/*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics.csv files found under {results_dir}")

    frames = [pd.read_csv(path) for path in metric_files]
    metrics = pd.concat(frames, ignore_index=True)
    for column in [
        "maximum_aggregate_magnitude",
        "aggregate_eq_error",
        "adjusted_coordinates",
        "total_coordinates",
    ]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics.to_csv(results_dir / "all_metrics.csv", index=False)

    last_rounds = metrics.groupby(["seed", "partition", "method"])["round"].transform("max")
    final = metrics[metrics["round"] > last_rounds - args.final_window].copy()

    per_seed = (
        final.groupby(["partition", "method", "seed"], as_index=False)
        .agg(
            final_accuracy=("test_accuracy", "mean"),
            final_loss=("test_loss", "mean"),
            max_aggregate_magnitude=("maximum_aggregate_magnitude", "max"),
            aggregate_eq_error=("aggregate_eq_error", "max"),
        )
    )
    by_method = (
        per_seed.groupby(["partition", "method"], as_index=False)
        .agg(
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            loss_mean=("final_loss", "mean"),
            loss_std=("final_loss", "std"),
            max_aggregate_magnitude=("max_aggregate_magnitude", "max"),
            aggregate_eq_error=("aggregate_eq_error", "max"),
        )
    )

    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(summary_dir / "per_seed_final_window.csv", index=False)
    by_method.to_csv(summary_dir / "summary_by_method.csv", index=False)

    quantized_source = "VMASK" if (metrics["method"] == "VMASK").any() else "Quantized SA"
    q_rows = metrics[metrics["method"] == quantized_source].copy()
    adj = (
        q_rows.groupby("partition")
        .agg(adjusted=("adjusted_coordinates", "sum"), total=("total_coordinates", "sum"))
        .reset_index()
    )
    adj["AdjRate"] = adj["adjusted"] / adj["total"]

    paper_rows = []
    for partition in sorted(metrics["partition"].unique()):
        row = {"Partition": partition}
        part_summary = by_method[by_method["partition"] == partition]
        for method, column in [
            ("FedAvg", "FedAvg Acc."),
            ("VMASK", "VMASK Acc."),
        ]:
            method_row = part_summary[part_summary["method"] == method]
            if method_row.empty:
                row[column] = ""
            else:
                item = method_row.iloc[0]
                row[column] = fmt_mean_std(item["accuracy_mean"], item["accuracy_std"])
        adj_row = adj[adj["partition"] == partition]
        row["AdjRate"] = "" if adj_row.empty else f"{100.0 * adj_row.iloc[0]['AdjRate']:.4f}%"
        vmask_row = part_summary[part_summary["method"] == "VMASK"]
        row["EqErr"] = "" if vmask_row.empty else f"{vmask_row.iloc[0]['aggregate_eq_error']:.3e}"
        fed_row = part_summary[part_summary["method"] == "FedAvg"]
        if fed_row.empty or vmask_row.empty:
            row["Acc. gap"] = ""
        else:
            gap = 100.0 * (vmask_row.iloc[0]["accuracy_mean"] - fed_row.iloc[0]["accuracy_mean"])
            row["Acc. gap"] = f"{gap:+.3f} pp"
        paper_rows.append(row)

    paper_table = pd.DataFrame(paper_rows)
    ordered = ["Partition", "FedAvg Acc.", "VMASK Acc.", "Acc. gap", "EqErr", "AdjRate"]
    paper_table = paper_table[ordered]
    paper_table.to_csv(summary_dir / "paper_table.csv", index=False)
    print(paper_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
