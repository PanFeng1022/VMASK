from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def isolate_conda_matplotlib() -> None:
    """Avoid mixing conda matplotlib with user-site mpl_toolkits on Windows."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    blocked = ("AppData\\Roaming\\Python", "AppData/Roaming/Python")
    sys.path[:] = [path for path in sys.path if not any(marker in path for marker in blocked)]
    for name in list(sys.modules):
        if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
            del sys.modules[name]


def configure_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
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


def load_grouped_metrics(results_dir: Path) -> pd.DataFrame:
    all_metrics = results_dir / "all_metrics.csv"
    metric_files = sorted(results_dir.glob("seed_*/*/metrics.csv"))
    if metric_files:
        metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    else:
        if all_metrics.exists():
            metrics = pd.read_csv(all_metrics)
        else:
            raise FileNotFoundError(f"No metrics.csv files found under {results_dir}")

    required = {"partition", "method", "round", "test_accuracy"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns in {results_dir}: {sorted(missing)}")
    if metrics.empty:
        raise ValueError(f"No metric records found under {results_dir}")

    return (
        metrics.groupby(["partition", "method", "round"], as_index=False)
        .agg(acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", "std"))
        .fillna(0.0)
        .sort_values(["partition", "method", "round"])
    )


def method_series(grouped: pd.DataFrame, partition: str) -> list[dict]:
    fedavg = grouped[(grouped["partition"] == partition) & (grouped["method"] == "FedAvg")].sort_values("round")
    vmask = grouped[(grouped["partition"] == partition) & (grouped["method"] == "VMASK")].sort_values("round")
    return [
        {
            "label": "FedAvg",
            "data": fedavg,
            "color": "#4E79A3",
            "linewidth": 1.70,
            "alpha": 0.98,
            "zorder": 2,
        },
        {
            "label": "VMASK",
            "data": vmask,
            "color": "#E53932",
            "linewidth": 1.48,
            "alpha": 0.58,
            "zorder": 3,
        },
    ]


def final_window_values(grouped: pd.DataFrame, partition: str, window_size: int = 10) -> dict[str, float]:
    max_round = int(grouped["round"].max())
    window = grouped[(grouped["partition"] == partition) & (grouped["round"] > max_round - window_size)]
    values: dict[str, float] = {}
    for method in ("FedAvg", "VMASK"):
        data = window[window["method"] == method]
        if not data.empty:
            values[method] = float(data["acc_mean"].mean()) * 100.0
    return values


def draw_accuracy_panel(
    ax,
    grouped: pd.DataFrame,
    partition: str,
    panel_label: str,
    y_max: float,
    show_legend: bool,
    show_values: bool,
) -> None:
    max_round = int(grouped["round"].max())
    for item in method_series(grouped, partition):
        data = item["data"]
        if data.empty:
            continue
        rounds = data["round"].to_numpy(dtype=float)
        mean = data["acc_mean"].to_numpy(dtype=float)
        std = data["acc_std"].to_numpy(dtype=float)
        ax.plot(
            rounds,
            mean,
            label=item["label"],
            color=item["color"],
            linewidth=item["linewidth"],
            alpha=item["alpha"],
            solid_joinstyle="miter",
            solid_capstyle="butt",
            zorder=item["zorder"],
        )
        ax.fill_between(rounds, mean - std, mean + std, color=item["color"], alpha=0.02, linewidth=0, zorder=1)

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
    ax.grid(True, linestyle="-", alpha=0.20)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.24, panel_label, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)

    if show_legend:
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(0.02, 0.50),
            frameon=True,
            framealpha=0.90,
            edgecolor="0.75",
            handlelength=2.5,
            borderpad=0.45,
        )
        legend.get_frame().set_linewidth(0.7)

    if show_values:
        values = final_window_values(grouped, partition)
        lines = []
        for method in ("FedAvg", "VMASK"):
            if method in values:
                lines.append(f"{method}: {values[method]:.2f}%")
        if lines:
            ax.text(
                0.98,
                0.06,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.8,
                linespacing=1.15,
                bbox={
                    "boxstyle": "square,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "0.76",
                    "alpha": 0.86,
                    "linewidth": 0.55,
                },
            )


def _yticks_for_limit(y_max: float) -> list[float]:
    if y_max <= 0.8:
        return [0.0, 0.2, 0.4, 0.6, 0.8]
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def add_zoom_inset(ax, grouped: pd.DataFrame, partition: str, y_max: float) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    zoom_end = int(grouped["round"].max())
    zoom_start = max(1, zoom_end - 49)
    zoom_values: list[float] = []
    for item in method_series(grouped, partition):
        window = item["data"][(item["data"]["round"] >= zoom_start) & (item["data"]["round"] <= zoom_end)]
        if not window.empty:
            zoom_values.extend(window["acc_mean"].tolist())
    if not zoom_values:
        return

    y_min = min(zoom_values)
    y_peak = max(zoom_values)
    margin = max(0.18 * (y_peak - y_min), 0.001)
    axins = inset_axes(ax, width="43%", height="38%", loc="center right", borderpad=0.70)

    for item in method_series(grouped, partition):
        window = item["data"][(item["data"]["round"] >= zoom_start) & (item["data"]["round"] <= zoom_end)]
        if window.empty:
            continue
        axins.plot(
            window["round"].to_numpy(dtype=float),
            window["acc_mean"].to_numpy(dtype=float),
            color=item["color"],
            linewidth=1.0,
            solid_joinstyle="miter",
            solid_capstyle="butt",
            zorder=item["zorder"],
        )

    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(max(0.0, y_min - margin), min(y_max, y_peak + margin))
    midpoint = zoom_start + (zoom_end - zoom_start) // 2
    axins.set_xticks([zoom_start, midpoint, zoom_end])
    axins.grid(True, linestyle=":", alpha=0.32, linewidth=0.5)
    axins.tick_params(axis="both", labelsize=7.0, length=2.0, pad=1.2)
    for spine in axins.spines.values():
        spine.set_edgecolor("0.35")
        spine.set_linewidth(0.7)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.58", lw=0.55)


def plot_single(grouped: pd.DataFrame, fig_dir: Path, output_stem: str, y_max: float, show_values: bool) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55), sharex=False, sharey=False)
    panels = [("iid", "(a) IID"), ("dirichlet", r"(b) Non-IID ($\alpha=0.5$)")]
    for index, (ax, (partition, panel_label)) in enumerate(zip(axes, panels)):
        draw_accuracy_panel(ax, grouped, partition, panel_label, y_max, show_legend=index == 0, show_values=show_values)
        add_zoom_inset(ax, grouped, partition, y_max)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.98, bottom=0.29, wspace=0.18)
    save_figure(fig, fig_dir, output_stem)


def plot_combined(
    mnist_grouped: pd.DataFrame,
    cifar_grouped: pd.DataFrame,
    fig_dir: Path,
    output_stem: str,
    show_values: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.55), sharex=False, sharey=False)
    panels = [
        (axes[0, 0], mnist_grouped, "iid", "(a) MNIST IID", 1.01, True),
        (axes[0, 1], mnist_grouped, "dirichlet", r"(b) MNIST Non-IID ($\alpha=0.5$)", 1.01, False),
        (axes[1, 0], cifar_grouped, "iid", "(c) CIFAR-10 IID", 0.8, False),
        (axes[1, 1], cifar_grouped, "dirichlet", r"(d) CIFAR-10 Non-IID ($\alpha=0.5$)", 0.8, False),
    ]
    for ax, grouped, partition, panel_label, y_max, show_legend in panels:
        draw_accuracy_panel(ax, grouped, partition, panel_label, y_max, show_legend, show_values=show_values)
        add_zoom_inset(ax, grouped, partition, y_max)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.13, wspace=0.18, hspace=0.43)
    save_figure(fig, fig_dir, output_stem)


def save_figure(fig, fig_dir: Path, output_stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / f"{output_stem}.png"
    pdf_path = fig_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benign utility accuracy curves")
    parser.add_argument("--results-dir", default="results/benign_mnist_300_seedcheck")
    parser.add_argument("--mnist-results-dir", default="results/benign_mnist_300_seedcheck")
    parser.add_argument("--cifar-results-dir", default="results/benign_cifar10_500")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-stem", default="fig4_benign_accuracy")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--show-values", action="store_true", help="Annotate panels with final-window accuracy values.")
    parser.add_argument("--y-max", type=float, default=1.01)
    args = parser.parse_args()

    isolate_conda_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style(plt)

    if args.combined:
        mnist_dir = Path(args.mnist_results_dir)
        cifar_dir = Path(args.cifar_results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else Path("results") / "benign_combined" / "figures"
        plot_combined(
            mnist_grouped=load_grouped_metrics(mnist_dir),
            cifar_grouped=load_grouped_metrics(cifar_dir),
            fig_dir=output_dir,
            output_stem=args.output_stem,
            show_values=args.show_values,
        )
    else:
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
        plot_single(
            grouped=load_grouped_metrics(results_dir),
            fig_dir=output_dir,
            output_stem=args.output_stem,
            y_max=args.y_max,
            show_values=args.show_values,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
