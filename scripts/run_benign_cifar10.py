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
    parser = argparse.ArgumentParser(description="Run benign CIFAR-10 VMASK utility experiment")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benign_cifar10.yaml"))
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument("--partitions", type=parse_str_list)
    parser.add_argument("--psi", type=float)
    parser.add_argument("--local-lr", type=float)
    parser.add_argument("--server-lr", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run one seed and one partition for 30 rounds")
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
    if args.psi is not None:
        cfg.psi = args.psi
    if args.local_lr is not None:
        cfg.local_lr = args.local_lr
    if args.server_lr is not None:
        cfg.server_lr = args.server_lr
    if args.overwrite:
        cfg.overwrite = True
    if args.skip_completed:
        cfg.skip_completed = True
    if args.quick:
        cfg.rounds = 30
        cfg.seeds = cfg.seeds[:1]
        cfg.partitions = cfg.partitions[:1]
        cfg.overwrite = True

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")
    print(f"Dataset/model: {cfg.dataset}/{cfg.model}")
    print(f"psi={cfg.psi} local_lr={cfg.local_lr} server_lr={cfg.server_lr}")
    run_benign_experiment(cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
