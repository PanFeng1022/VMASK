from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmask_exp.benign import run_benign_experiment
from vmask_exp.config import load_config, resolve_device


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benign MNIST VMASK utility experiment")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benign_mnist.yaml"))
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument("--partitions", type=parse_str_list)
    parser.add_argument("--protocol-execution", action="store_true")
    parser.add_argument("--protocol-profile", choices=["p80", "p100"])
    parser.add_argument("--protocol-committee-size", type=int)
    parser.add_argument("--protocol-threshold", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run one seed for 30 rounds")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.results_dir:
        cfg.results_dir = args.results_dir
    if args.download:
        cfg.download = True
    if args.device:
        cfg.device = args.device
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.partitions is not None:
        cfg.partitions = args.partitions
    if args.protocol_execution:
        cfg.protocol_execution = True
    if args.protocol_profile is not None:
        cfg.protocol_profile = args.protocol_profile
    if args.protocol_committee_size is not None:
        cfg.protocol_committee_size = args.protocol_committee_size
    if args.protocol_threshold is not None:
        cfg.protocol_threshold = args.protocol_threshold
    if args.overwrite:
        cfg.overwrite = True
    if args.quick:
        cfg.rounds = 30
        cfg.seeds = cfg.seeds[:1]
        cfg.overwrite = True

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")
    run_benign_experiment(cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
