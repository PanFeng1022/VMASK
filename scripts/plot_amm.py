
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def isolate_conda_matplotlib() -> None:
    """Avoid mixing conda matplotlib with user-site mpl_toolkits on Windows."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    blocked = ("AppData\\Roaming\\Python", "AppData/Roaming/Python")
    sys.path[:] = [
        path
        for path in sys.path
        if not any(marker in path for marker in blocked)
    ]

    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]


def parse_ratios(raw: str) -> list[float]:
    return [
        float(value.strip())
        for value in raw.split(",")
        if value.strip()
    ]


def configure_style(plt) -> None:
    """Use the same visual style as the benign-utility figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.45,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


STYLE_META: dict[float, dict[str, object]] = {
    0.10: {
        "label": r"$\gamma=0.1$",
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.60,
        "zorder": 2,
    },
    0.20: {
        "label": r"$\gamma=0.2$",
        "color": "#E69F00",
        "linestyle": "-",
        "linewidth": 1.60,
        "zorder": 3,
    },
    0.30: {
        "label": r"$\gamma=0.3$",
        "color": "#8E44AD",
        "linestyle": "-",
        "linewidth": 1.60,
        "zorder": 4,
    },
}


def load_amm_metrics(results_dir: Path) -> pd.DataFrame:
    all_metrics = results_dir / "all_metrics.csv"
    metric_files = sorted(results_dir.glob("gamma_*/*/seed_*/metrics.csv"))
    if metric_files:
        metrics = pd.concat(
            [pd.read_csv(path) for path in metric_files],
            ignore_index=True,
        )
    elif all_metrics.exists():
        metrics = pd.read_csv(all_metrics)
    else:
        raise FileNotFoundError(f"No AMM metrics found under {results_dir}")

    required_columns = {
        "seed",
        "gamma",
        "mode",
        "round",
        "test_accuracy",
        "agg_err",
    }
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return metrics


def select_curve(
    grouped: pd.DataFrame,
    mode: str,
    gamma: float,
) -> pd.DataFrame:
    mask = (
        (grouped["mode"] == mode)
        & np.isclose(
            grouped["gamma"].to_numpy(dtype=float),
            gamma,
            rtol=0.0,
            atol=1e-9,
        )
    )
    return grouped.loc[mask].sort_values("round")


def plot_accuracy_figure(
    grouped: pd.DataFrame,
    metrics: pd.DataFrame,
    curve_gammas: list[float],
    fig_dir: Path,
    output_stem: str,
    zoom_start: int,
    zoom_end: int,
    y_max: float,
) -> None:
    import matplotlib.pyplot as plt

    # Match the benign figure width while retaining the original AMM height.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 2.55),
        sharex=False,
        sharey=False,
    )

    panels = [
        ("verification_off", "(a) Without verification"),
        ("vmask", "(b) With VMASK"),
    ]

    max_round = int(metrics["round"].max())

    for index, (ax, (mode, panel_label)) in enumerate(
        zip(axes, panels)
    ):
        draw_accuracy_panel(
            ax=ax,
            grouped=grouped,
            mode=mode,
            panel_label=panel_label,
            curve_gammas=curve_gammas,
            max_round=max_round,
            y_max=y_max,
        )

        if index == 0:
            legend = ax.legend(
                loc="center left",
                frameon=True,
                framealpha=0.90,
                edgecolor="0.75",
                handlelength=2.5,
                borderpad=0.45,
            )
            legend.get_frame().set_linewidth(0.7)

    add_zoom_inset(
        ax=axes[1],
        grouped=grouped,
        curve_gammas=curve_gammas,
        zoom_start=zoom_start,
        zoom_end=min(zoom_end, max_round),
    )

    # More bottom space is needed because the figure height remains 2.55.
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.98,
        bottom=0.29,
        wspace=0.18,
    )

    png_path = fig_dir / f"{output_stem}.png"
    pdf_path = fig_dir / f"{output_stem}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def draw_accuracy_panel(
    ax,
    grouped: pd.DataFrame,
    mode: str,
    panel_label: str,
    curve_gammas: list[float],
    max_round: int,
    y_max: float,
) -> None:
    for gamma in curve_gammas:
        data = select_curve(
            grouped=grouped,
            mode=mode,
            gamma=gamma,
        )
        if data.empty:
            continue

        meta = STYLE_META.get(
            round(gamma, 2),
            {
                "label": rf"$\gamma={gamma:.1f}$",
                "color": "#333333",
                "linestyle": "-",
                "linewidth": 1.45,
                "zorder": 2,
            },
        )

        rounds = data["round"].to_numpy(dtype=float)
        mean = data["acc_mean"].to_numpy(dtype=float)
        std = data["acc_std"].to_numpy(dtype=float)

        # Directly plot per-round means. No smoothing is applied.
        ax.plot(
            rounds,
            mean,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=meta["linewidth"],
            solid_joinstyle="miter",
            solid_capstyle="butt",
            zorder=meta["zorder"],
        )

        # Retain the same subtle uncertainty band as the benign figure.
        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=meta["color"],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )

    if mode == "verification_off":
        ax.axhline(
            0.10,
            color="0.45",
            linewidth=0.55,
            linestyle=":",
            alpha=0.75,
            zorder=0,
        )

    ax.set_xlabel("Round", labelpad=2)
    ax.set_ylabel("Accuracy", labelpad=2)
    ax.set_xlim(0, max_round)
    ax.set_ylim(0.0, y_max)
    ax.set_yticks(_yticks_for_limit(y_max))

    if max_round >= 500:
        ax.set_xticks([0, 100, 200, 300, 400, 500])
    elif max_round >= 300:
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    elif max_round >= 200:
        ax.set_xticks([0, 50, 100, 150, 200])

    ax.grid(
        True,
        linestyle="-",
        alpha=0.20,
    )
    ax.set_axisbelow(True)

    ax.text(
        0.5,
        -0.27,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)


def _yticks_for_limit(y_max: float) -> list[float]:
    if y_max <= 0.8:
        return [0.0, 0.2, 0.4, 0.6, 0.8]
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _default_y_max(results_dir: Path) -> float:
    return 0.8 if "cifar" in str(results_dir).lower() else 1.01


def add_zoom_inset(
    ax,
    grouped: pd.DataFrame,
    curve_gammas: list[float],
    zoom_start: int,
    zoom_end: int,
) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import (
        inset_axes,
        mark_inset,
    )

    if zoom_end <= zoom_start:
        return

    zoom_values: list[float] = []

    for gamma in curve_gammas:
        data = select_curve(
            grouped=grouped,
            mode="vmask",
            gamma=gamma,
        )
        window = data[
            (data["round"] >= zoom_start)
            & (data["round"] <= zoom_end)
        ]
        if not window.empty:
            zoom_values.extend(window["acc_mean"].tolist())

    if not zoom_values:
        return

    y_min = min(zoom_values)
    y_max = max(zoom_values)
    margin = max(
        0.15 * (y_max - y_min),
        0.0004,
    )

    axins = inset_axes(
        ax,
        width="42%",
        height="38%",
        loc="center right",
        borderpad=0.70,
    )

    for gamma in curve_gammas:
        data = select_curve(
            grouped=grouped,
            mode="vmask",
            gamma=gamma,
        )
        window = data[
            (data["round"] >= zoom_start)
            & (data["round"] <= zoom_end)
        ]
        if window.empty:
            continue

        meta = STYLE_META.get(
            round(gamma, 2),
            {
                "color": "#333333",
                "linestyle": "-",
                "zorder": 2,
            },
        )

        # Directly plot unsmoothed per-round means.
        axins.plot(
            window["round"].to_numpy(dtype=float),
            window["acc_mean"].to_numpy(dtype=float),
            color=meta["color"],
            linestyle="-",
            linewidth=1.0,
            solid_joinstyle="miter",
            solid_capstyle="butt",
            zorder=meta["zorder"],
        )

    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(
        max(0.0, y_min - margin),
        min(1.0, y_max + margin),
    )
    axins.grid(
        True,
        linestyle=":",
        alpha=0.32,
        linewidth=0.5,
    )
    axins.tick_params(
        axis="both",
        labelsize=7.2,
        pad=1.2,
    )

    for spine in axins.spines.values():
        spine.set_edgecolor("0.35")
        spine.set_linewidth(0.7)

    mark_inset(
        ax,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.58",
        lw=0.55,
    )


def write_accuracy_table(
    summary: pd.DataFrame,
    table_gammas: list[float],
    summary_dir: Path,
) -> None:
    acc_table = (
        summary[
            summary["gamma"].apply(
                lambda value: any(
                    np.isclose(
                        float(value),
                        gamma,
                        rtol=0.0,
                        atol=1e-9,
                    )
                    for gamma in table_gammas
                )
            )
        ]
        .set_index(["gamma", "mode"])
        .sort_index()
    )

    rows: list[dict[str, object]] = []

    for gamma in table_gammas:
        row: dict[str, object] = {
            "gamma": f"{100.0 * gamma:.0f}\\%",
        }

        for mode, label in [
            ("verification_off", "verification_disabled"),
            ("vmask", "vmask"),
        ]:
            matching_index = next(
                (
                    index
                    for index in acc_table.index
                    if index[1] == mode
                    and np.isclose(
                        float(index[0]),
                        gamma,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ),
                None,
            )

            if matching_index is not None:
                item = acc_table.loc[matching_index]
                mean = float(item["acc_mean"]) * 100.0
                std = float(item["acc_std"]) * 100.0

                row[f"{label}_accuracy_mean"] = mean
                row[f"{label}_accuracy_std"] = std
                row[f"{label}_accuracy"] = (
                    f"{mean:.2f} +/- {std:.2f}"
                )
            else:
                row[f"{label}_accuracy_mean"] = float("nan")
                row[f"{label}_accuracy_std"] = float("nan")
                row[f"{label}_accuracy"] = "NA"

        off_mean = row["verification_disabled_accuracy_mean"]
        vmask_mean = row["vmask_accuracy_mean"]

        if pd.notna(off_mean) and pd.notna(vmask_mean):
            gap = float(vmask_mean) - float(off_mean)
            row["accuracy_gap_pp"] = gap
            row["accuracy_gap"] = f"{gap:.2f}"
        else:
            row["accuracy_gap_pp"] = float("nan")
            row["accuracy_gap"] = "NA"

        rows.append(row)

    output = pd.DataFrame(rows)
    csv_path = summary_dir / "amm_accuracy_table.csv"
    tex_path = summary_dir / "amm_accuracy_table.tex"

    output.to_csv(
        csv_path,
        index=False,
    )

    table_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\gamma$ & Verification disabled & VMASK & Accuracy gap \\",
        r"\midrule",
    ]

    for row in rows:
        off_acc = str(
            row["verification_disabled_accuracy"]
        ).replace(
            " +/- ",
            r"\pm",
        )
        vmask_acc = str(
            row["vmask_accuracy"]
        ).replace(
            " +/- ",
            r"\pm",
        )

        table_lines.append(
            rf"{row['gamma']} & "
            rf"${off_acc}$ & "
            rf"${vmask_acc}$ & "
            rf"${row['accuracy_gap']}$ pp \\"
        )

    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )

    tex_path.write_text(
        "\n".join(table_lines),
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


def grouped_accuracy(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["mode", "gamma", "round"],
            as_index=False,
        )
        .agg(
            acc_mean=("test_accuracy", "mean"),
            acc_std=("test_accuracy", "std"),
        )
        .fillna(0.0)
        .sort_values(["mode", "gamma", "round"])
    )


def plot_combined_accuracy_figure(
    mnist_metrics: pd.DataFrame,
    cifar_metrics: pd.DataFrame,
    curve_gammas: list[float],
    fig_dir: Path,
    output_stem: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 5.55),
        sharex=False,
        sharey=False,
    )

    panels = [
        (axes[0, 0], grouped_accuracy(mnist_metrics), mnist_metrics, "verification_off", "(a) MNIST Without verification", 1.01),
        (axes[0, 1], grouped_accuracy(mnist_metrics), mnist_metrics, "vmask", "(b) MNIST With VMASK", 1.01),
        (axes[1, 0], grouped_accuracy(cifar_metrics), cifar_metrics, "verification_off", "(c) CIFAR-10 Without verification", 0.8),
        (axes[1, 1], grouped_accuracy(cifar_metrics), cifar_metrics, "vmask", "(d) CIFAR-10 With VMASK", 0.8),
    ]

    for index, (ax, grouped, metrics, mode, panel_label, y_max) in enumerate(panels):
        max_round = int(metrics["round"].max())
        draw_accuracy_panel(
            ax=ax,
            grouped=grouped,
            mode=mode,
            panel_label=panel_label,
            curve_gammas=curve_gammas,
            max_round=max_round,
            y_max=y_max,
        )
        if index == 0:
            legend = ax.legend(
                loc="center left",
                frameon=True,
                framealpha=0.90,
                edgecolor="0.75",
                handlelength=2.5,
                borderpad=0.45,
            )
            legend.get_frame().set_linewidth(0.7)
        if mode == "vmask":
            add_zoom_inset(
                ax=ax,
                grouped=grouped,
                curve_gammas=curve_gammas,
                zoom_start=max(1, max_round - 50),
                zoom_end=max_round,
            )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.13, wspace=0.18, hspace=0.43)

    png_path = fig_dir / f"{output_stem}.png"
    pdf_path = fig_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot AMM effectiveness"
    )
    parser.add_argument(
        "--results-dir",
        default="results/amm_mnist_arithmetic_mismatch",
    )
    parser.add_argument(
        "--mnist-results-dir",
        default="results/amm_mnist_arithmetic_mismatch",
    )
    parser.add_argument(
        "--cifar-results-dir",
        default="results/amm_cifar10_arithmetic_mismatch_500",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--combined",
        action="store_true",
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-stem",
        default="fig5_amm_accuracy",
    )
    parser.add_argument(
        "--curve-ratios",
        default="0.10,0.20,0.30",
    )
    parser.add_argument(
        "--table-ratios",
        default="0.10,0.20,0.30",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--zoom-end",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = load_amm_metrics(results_dir)

    max_round = int(metrics["round"].max())
    zoom_end = args.zoom_end if args.zoom_end is not None else max_round
    zoom_start = args.zoom_start if args.zoom_start is not None else max(1, zoom_end - 50)

    last_round = metrics.groupby(
        ["seed", "gamma", "mode"]
    )["round"].transform("max")

    final = metrics[
        metrics["round"] > last_round - args.final_window
    ].copy()

    summary = (
        final.groupby(
            ["gamma", "mode", "seed"],
            as_index=False,
        )
        .agg(
            final_accuracy=("test_accuracy", "mean"),
            final_agg_err=("agg_err", "mean"),
        )
        .groupby(
            ["gamma", "mode"],
            as_index=False,
        )
        .agg(
            acc_mean=("final_accuracy", "mean"),
            acc_std=("final_accuracy", "std"),
            err_mean=("final_agg_err", "mean"),
            err_std=("final_agg_err", "std"),
        )
        .fillna(0.0)
    )

    grouped = (
        metrics.groupby(
            ["mode", "gamma", "round"],
            as_index=False,
        )
        .agg(
            acc_mean=("test_accuracy", "mean"),
            acc_std=("test_accuracy", "std"),
        )
        .fillna(0.0)
        .sort_values(["mode", "gamma", "round"])
    )

    curve_gammas = parse_ratios(args.curve_ratios)
    table_gammas = parse_ratios(args.table_ratios)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dir = results_dir / "summary"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    isolate_conda_matplotlib()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style(plt)

    if args.combined:
        output_dir = Path(args.output_dir) if args.output_dir else Path("results") / "amm_combined" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_combined_accuracy_figure(
            mnist_metrics=load_amm_metrics(Path(args.mnist_results_dir)),
            cifar_metrics=load_amm_metrics(Path(args.cifar_results_dir)),
            curve_gammas=curve_gammas,
            fig_dir=output_dir,
            output_stem=args.output_stem,
        )
        return 0

    plot_accuracy_figure(
        grouped=grouped,
        metrics=metrics,
        curve_gammas=curve_gammas,
        fig_dir=fig_dir,
        output_stem=args.output_stem,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        y_max=args.y_max if args.y_max is not None else _default_y_max(results_dir),
    )

    write_accuracy_table(
        summary=summary,
        table_gammas=table_gammas,
        summary_dir=summary_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
