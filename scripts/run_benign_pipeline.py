from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "benign_mnist.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the benign utility pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir")
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--seeds", help="Optional comma-separated seed subset.")
    parser.add_argument("--partitions", help="Optional comma-separated partition subset.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    rounds = int(cfg["rounds"])
    seeds = parse_int_list(args.seeds) if args.seeds else list(cfg["seeds"])
    partitions = parse_str_list(args.partitions) if args.partitions else list(cfg["partitions"])

    results_dir = Path(args.results_dir or cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    print("Benign utility pipeline")
    print(f"  config      : {config_path}")
    print(f"  results_dir : {results_dir}")
    print(f"  rounds      : {rounds}")
    print(f"  seeds       : {seeds}")
    print(f"  partitions  : {partitions}")
    print("")

    if not args.skip_run:
        for seed in seeds:
            for partition in partitions:
                metrics_path = metrics_file(results_dir, seed, partition)
                complete = metrics_complete(metrics_path, rounds)
                if complete and not args.force:
                    print(f"[skip] complete: seed={seed} partition={partition}")
                    continue

                runner = "run_benign_cifar10.py" if str(cfg["dataset"]).lower() == "cifar10" else "run_benign_mnist.py"
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / runner),
                    "--config",
                    str(config_path),
                    "--seeds",
                    str(seed),
                    "--partitions",
                    partition,
                    "--results-dir",
                    str(results_dir),
                ]
                if args.force or metrics_path.exists():
                    command.append("--overwrite")
                run_command(command, args.dry_run)

    if not args.skip_summary:
        run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "summarize_benign.py"),
                "--results-dir",
                str(results_dir),
                "--final-window",
                str(args.final_window),
            ],
            args.dry_run,
        )

    if not args.skip_plot:
        run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_benign.py"),
                "--results-dir",
                str(results_dir),
            ],
            args.dry_run,
        )

    return 0


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / "src"))
        from vmask_exp.config import _load_simple_yaml

        data = _load_simple_yaml(text)
    return data


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def metrics_file(results_dir: Path, seed: int, partition: str) -> Path:
    return results_dir / f"seed_{seed}" / partition / "metrics.csv"


def metrics_complete(path: Path, rounds: int) -> bool:
    if not path.exists():
        return False

    row_count = 0
    last_round: int | None = None
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_count += 1
                last_round = int(float(row["round"]))
    except (OSError, KeyError, ValueError):
        return False

    # Benign metrics contain one FedAvg row and one VMASK row per round.
    return row_count == 2 * rounds and last_round == rounds


def run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(quote_part(part) for part in command)
    print(f"[run] {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def quote_part(part: str) -> str:
    if any(ch.isspace() for ch in part):
        return f'"{part}"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
