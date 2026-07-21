from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import math


def mode_label(mode: str) -> str:
    return "Verification disabled" if mode == "verification_off" else "VMASK"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize AMM effectiveness runs")
    parser.add_argument("--results-dir", default="results/amm_mnist_arithmetic_mismatch")
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--aggerr-ratios", default="0.10,0.20,0.30")
    args = parser.parse_args()
    aggerr_table_ratios = {
        round(float(x.strip()), 10)
        for x in args.aggerr_ratios.split(",")
        if x.strip()
    }

    results_dir = Path(args.results_dir)
    metric_files = sorted(results_dir.glob("gamma_*/*/seed_*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No AMM metrics found under {results_dir}")

    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics["mode_label"] = metrics["mode"].map(mode_label)
    metrics.to_csv(results_dir / "all_metrics.csv", index=False)

    last_round = metrics.groupby(["seed", "gamma", "mode"])["round"].transform("max")
    final = metrics[metrics["round"] > last_round - args.final_window].copy()
    per_seed_aggs = {
        "final_accuracy": ("test_accuracy", "mean"),
        "final_loss": ("test_loss", "mean"),
        "mean_agg_err": ("agg_err", "mean"),
        "rejected_clients": ("rejected_clients", "max"),
    }
    if {"amm_reject_rate", "benign_accept_rate"}.issubset(final.columns):
        per_seed_aggs.update(
            {
                "amm_reject_rate": ("amm_reject_rate", "mean"),
                "benign_accept_rate": ("benign_accept_rate", "mean"),
            }
        )
    per_seed = (
        final.groupby(["gamma", "mode", "mode_label", "seed"], as_index=False)
        .agg(**per_seed_aggs)
    )
    summary_aggs = {
        "accuracy_mean": ("final_accuracy", "mean"),
        "accuracy_std": ("final_accuracy", "std"),
        "agg_err_mean": ("mean_agg_err", "mean"),
        "agg_err_std": ("mean_agg_err", "std"),
        "rejected_clients": ("rejected_clients", "max"),
    }
    if {"amm_reject_rate", "benign_accept_rate"}.issubset(per_seed.columns):
        summary_aggs.update(
            {
                "amm_reject_rate": ("amm_reject_rate", "mean"),
                "benign_accept_rate": ("benign_accept_rate", "mean"),
            }
        )
    summary = (
        per_seed.groupby(["gamma", "mode", "mode_label"], as_index=False)
        .agg(**summary_aggs)
    )
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(summary_dir / "per_seed_final_window.csv", index=False)
    summary.to_csv(summary_dir / "summary_by_gamma.csv", index=False)

    if {"agg_err_num_sq", "agg_err_den_sq"}.issubset(metrics.columns):
        cum_source = metrics[metrics["mode"] == "verification_off"].copy()
        cum_per_seed = (
            cum_source.groupby(["gamma", "seed"], as_index=False)
            .agg(
                num_sq_sum=("agg_err_num_sq", "sum"),
                den_sq_sum=("agg_err_den_sq", "sum"),
            )
        )
        eps = 1e-12
        cum_per_seed["cumulative_agg_err"] = cum_per_seed.apply(
            lambda r: math.sqrt(float(r["num_sq_sum"])) / (math.sqrt(float(r["den_sq_sum"])) + eps),
            axis=1,
        )
        cum_summary = (
            cum_per_seed.groupby("gamma", as_index=False)
            .agg(
                cumulative_agg_err_mean=("cumulative_agg_err", "mean"),
                cumulative_agg_err_std=("cumulative_agg_err", "std"),
            )
            .fillna(0.0)
        )
        base_rows = cum_summary[cum_summary["gamma"].round(10) == 0.10]
        base = float(base_rows["cumulative_agg_err_mean"].iloc[0]) if not base_rows.empty else float("nan")
        if base > 0:
            cum_summary["relative_growth"] = cum_summary["cumulative_agg_err_mean"] / base
        else:
            cum_summary["relative_growth"] = float("nan")
        cum_per_seed.to_csv(summary_dir / "cumulative_aggerr_per_seed.csv", index=False)
        cum_summary.to_csv(summary_dir / "cumulative_aggerr_summary.csv", index=False)

        table_lines = [
            r"\begin{tabular}{ccc}",
            r"\toprule",
            r"$\gamma$ & $\mathrm{AggErr}_{\gamma}$ & Ratio to $\gamma=0.1$ \\",
            r"\midrule",
        ]
        for _, row in cum_summary.sort_values("gamma").iterrows():
            if round(float(row["gamma"]), 10) not in aggerr_table_ratios:
                continue
            gamma_label = f"{100 * float(row['gamma']):.0f}\\%"
            growth = "TBD" if pd.isna(row["relative_growth"]) else f"{float(row['relative_growth']):.2f}\\times"
            table_lines.append(
                rf"${gamma_label}$ & "
                rf"${float(row['cumulative_agg_err_mean']):.4f}\pm{float(row['cumulative_agg_err_std']):.4f}$ & "
                rf"${growth}$ \\"
            )
        table_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
        (summary_dir / "cumulative_aggerr_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    table = summary.copy()
    table["gamma"] = table["gamma"].map(lambda x: f"{100*x:.0f}%")
    table["accuracy"] = table.apply(
        lambda r: f"{100*r['accuracy_mean']:.2f} +/- {0 if pd.isna(r['accuracy_std']) else 100*r['accuracy_std']:.2f}",
        axis=1,
    )
    table["AggErr"] = table.apply(
        lambda r: f"{r['agg_err_mean']:.4e} +/- {0 if pd.isna(r['agg_err_std']) else r['agg_err_std']:.4e}",
        axis=1,
    )
    if {"amm_reject_rate", "benign_accept_rate"}.issubset(table.columns):
        table["AMMRejectRate"] = table["amm_reject_rate"].map(lambda x: f"{100*x:.2f}%")
        table["BenignAcceptRate"] = table["benign_accept_rate"].map(lambda x: f"{100*x:.2f}%")
        paper = table[
            [
                "gamma",
                "mode_label",
                "accuracy",
                "AggErr",
                "rejected_clients",
                "AMMRejectRate",
                "BenignAcceptRate",
            ]
        ]
    else:
        paper = table[["gamma", "mode_label", "accuracy", "AggErr", "rejected_clients"]]
    paper.to_csv(summary_dir / "paper_table.csv", index=False)
    print(paper.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
