from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmask_exp.config import resolve_device, set_global_seed
from vmask_exp.data import (
    dataset_labels,
    load_dataset,
    partition_dataset,
    partition_seed,
    save_label_distribution,
)
from vmask_exp.model import assert_parameter_count, create_model
from vmask_exp.params import add_update_
from vmask_exp.protocol import VMaskProtocol
from vmask_exp.protocol.config import profile_p100, profile_p80
from vmask_exp.quantization import (
    aggregate_update_from_qsum,
    check_modulus,
    stochastic_quantize,
)
from vmask_exp.trainer import evaluate_model, train_client


@dataclass
class AmmConfig:
    experiment_name: str = "amm_mnist_arithmetic_mismatch"
    data_dir: str = "data"
    dataset: str = "mnist"
    model: str = "mnist_cnn"
    results_dir: str = "results/amm_mnist_arithmetic_mismatch"
    download: bool = False
    device: str = "auto"
    overwrite: bool = False
    seeds: list[int] = field(default_factory=lambda: [8207, 9090, 9421, 7515, 4407])
    malicious_ratios: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    num_clients: int = 20
    rounds: int = 300
    local_steps: int = 20
    batch_size: int = 64
    local_lr: float = 0.05
    momentum: float = 0.9
    server_lr: float = 0.1
    psi: float = 256.0
    amax: int = 15
    modulus: int = 2**128
    expected_dimension: int = 27_562
    r0: int = 1023
    mask_limbs: int = 10
    protocol_execution: bool = True
    vmask_backend: str = "full"
    protocol_profile: str = "p80"
    protocol_committee_size: int = 5
    protocol_threshold: int = 3
    protocol_kappa: int = 14
    protocol_modulus: int = 2**128
    attack_vector_seed: int = 20260617
    attack_limb0_abs: int = 16
    eval_batch_size: int = 512
    num_workers: int = 0
    skip_completed: bool = False
    checkpoint_every: int = 1


FIELDS = [
    "dataset",
    "partition",
    "dirichlet_alpha",
    "seed",
    "gamma",
    "malicious_clients",
    "mode",
    "round",
    "test_accuracy",
    "test_loss",
    "mean_local_loss",
    "accepted_clients",
    "rejected_clients",
    "amm_reject_rate",
    "malicious_submissions_constructed",
    "malicious_submissions_rejected",
    "malicious_submissions_accepted",
    "benign_accept_rate",
    "agg_err",
    "agg_err_num_sq",
    "agg_err_den_sq",
    "max_aggregate_magnitude",
    "attack_residual_l2_sq",
    "attack_limb0_abs",
    "attack_vector_seed",
    "protocol_execution",
    "relation_failures",
    "aggregate_equality",
]


def load_amm_config(path: str | Path | None) -> AmmConfig:
    from vmask_exp.config import _load_simple_yaml

    cfg = AmmConfig()
    if path is None:
        return cfg
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    for key, value in data.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AMM effectiveness experiment")
    parser.add_argument("--config", default=str(ROOT / "configs" / "amm_mnist.yaml"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--ratios")
    parser.add_argument("--modes", default="verification_off,vmask")
    parser.add_argument("--partition", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--psi", type=float)
    parser.add_argument("--local-lr", type=float)
    parser.add_argument("--server-lr", type=float)
    parser.add_argument("--protocol-execution", action="store_true")
    parser.add_argument("--protocol-profile", choices=["p80", "p100"])
    parser.add_argument("--protocol-committee-size", type=int)
    parser.add_argument("--protocol-threshold", type=int)
    parser.add_argument("--protocol-kappa", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_amm_config(args.config)
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.seeds:
        cfg.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.ratios:
        cfg.malicious_ratios = [
            float(x.strip()) for x in args.ratios.split(",") if x.strip()
        ]
    if args.partition is not None:
        cfg.partition = args.partition
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = args.dirichlet_alpha
    if args.psi is not None:
        cfg.psi = args.psi
    if args.local_lr is not None:
        cfg.local_lr = args.local_lr
    if args.server_lr is not None:
        cfg.server_lr = args.server_lr
    if args.protocol_execution:
        cfg.protocol_execution = True
    if args.protocol_profile is not None:
        cfg.protocol_profile = args.protocol_profile
    if args.protocol_committee_size is not None:
        cfg.protocol_committee_size = args.protocol_committee_size
    if args.protocol_threshold is not None:
        cfg.protocol_threshold = args.protocol_threshold
    if args.protocol_kappa is not None:
        cfg.protocol_kappa = args.protocol_kappa
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    invalid_modes = sorted(set(modes) - {"verification_off", "vmask"})
    if invalid_modes:
        raise ValueError(f"Unsupported AMM mode(s): {invalid_modes}")
    if cfg.vmask_backend != "full":
        raise ValueError(
            "The public AMM runner supports only vmask_backend=full"
        )
    if "vmask" in modes and not cfg.protocol_execution:
        raise ValueError(
            "VMASK mode requires protocol_execution=true"
        )
    if args.results_dir:
        cfg.results_dir = args.results_dir
    if args.download:
        cfg.download = True
    if args.overwrite:
        cfg.overwrite = True
    if args.skip_completed:
        cfg.skip_completed = True
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.quick:
        cfg.rounds = 5
        cfg.seeds = cfg.seeds[:1]
        cfg.malicious_ratios = cfg.malicious_ratios[:1]
        cfg.overwrite = True

    if cfg.partition not in {"iid", "dirichlet"}:
        raise ValueError(f"Unsupported partition: {cfg.partition}")
    if cfg.dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    if cfg.rounds <= 0:
        raise ValueError("rounds must be positive")
    if cfg.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    check_modulus(cfg.modulus, cfg.num_clients, cfg.amax)
    validate_arithmetic_mismatch_attack(cfg)
    device = resolve_device(cfg.device)
    train_dataset, test_dataset = load_dataset(cfg.dataset, cfg.data_dir, cfg.download)
    labels = dataset_labels(train_dataset)

    for seed in cfg.seeds:
        current_partition_seed = (
            partition_seed(seed, cfg.partition)
            if cfg.partition == "dirichlet"
            else _stable_seed(seed, 0, 0, 19)
        )
        partition = partition_dataset(
            labels,
            cfg.num_clients,
            cfg.partition,
            current_partition_seed,
            alpha=cfg.dirichlet_alpha,
        )
        for gamma in cfg.malicious_ratios:
            malicious_count = int(round(gamma * cfg.num_clients))
            malicious_ids = list(range(malicious_count))
            for mode in modes:
                run_one(
                    cfg,
                    seed,
                    gamma,
                    malicious_ids,
                    mode,
                    partition,
                    current_partition_seed,
                    train_dataset,
                    test_dataset,
                    device,
                )
    return 0


def run_one(
    cfg,
    seed,
    gamma,
    malicious_ids,
    mode,
    partition,
    current_partition_seed,
    train_dataset,
    test_dataset,
    device,
) -> None:
    import numpy as np
    import torch

    started = time.perf_counter()
    gamma_tag = str(gamma).replace(".", "p")
    out_dir = Path(cfg.results_dir) / f"gamma_{gamma_tag}" / mode / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    config_path = out_dir / "config.json"
    label_distribution_path = out_dir / "label_distribution.csv"
    summary_path = out_dir / "run_summary.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    checkpoint_tmp_path = out_dir / "checkpoint.pt.tmp"
    partition_digest = _partition_digest(partition.client_indices)
    run_config = {
        **asdict(cfg),
        "seed": seed,
        "partition_seed": current_partition_seed,
        "partition_digest": partition_digest,
        "resolved_device": str(device),
        "gamma": gamma,
        "mode": mode,
        "malicious_ids": malicious_ids,
    }
    run_identity = _run_identity(run_config)
    resume_requested = False

    if cfg.overwrite:
        for stale in [
            metrics_path,
            config_path,
            label_distribution_path,
            summary_path,
            checkpoint_path,
            checkpoint_tmp_path,
        ]:
            if stale.exists():
                stale.unlink()
    elif metrics_path.exists():
        if (
            cfg.skip_completed
            and _metrics_complete(metrics_path, cfg.rounds)
            and summary_path.exists()
        ):
            print(
                f"skip completed AMM run: seed={seed} gamma={gamma:.2f} mode={mode} rounds={cfg.rounds}"
            )
            return
        if cfg.skip_completed:
            if checkpoint_path.exists():
                resume_requested = True
            else:
                print(
                    f"restart interrupted AMM run without checkpoint: seed={seed} "
                    f"gamma={gamma:.2f} mode={mode} rounds={cfg.rounds}"
                )
                for stale in [
                    metrics_path,
                    config_path,
                    label_distribution_path,
                    summary_path,
                    checkpoint_tmp_path,
                ]:
                    if stale.exists():
                        stale.unlink()
        else:
            raise FileExistsError(
                f"{metrics_path} exists but is not marked for overwrite. "
                "Use --skip-completed to resume it or --overwrite to replace it."
            )
    elif checkpoint_path.exists():
        raise RuntimeError(
            f"{checkpoint_path} exists without {metrics_path}; refusing to resume "
            "because the saved round sequence does not match the configured run"
        )

    checkpoint = None
    if resume_requested:
        checkpoint = _load_checkpoint(checkpoint_path, torch)
        _validate_checkpoint(checkpoint, run_identity, cfg.rounds)

    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_label_distribution(label_distribution_path, partition.label_counts)

    set_global_seed(seed)
    model = create_model(cfg.model).to(device)
    assert_parameter_count(model, cfg.expected_dimension)
    attack_residual = build_arithmetic_mismatch_residual(cfg)
    attack_direction = [
        int(v) // (2 * int(cfg.attack_limb0_abs)) for v in attack_residual.tolist()
    ]
    protocol = (
        build_protocol(cfg)
        if cfg.protocol_execution and cfg.vmask_backend == "full"
        else None
    )
    malicious = set(malicious_ids)
    honest_indices = [
        client_id for client_id in range(cfg.num_clients) if client_id not in malicious
    ]
    if mode == "vmask" and not honest_indices:
        raise ValueError("VMASK mode requires at least one accepted client")

    start_round = 1
    elapsed_before_resume = 0.0
    cumulative_num_sq = 0.0
    cumulative_den_sq = 0.0
    malicious_constructed_total = 0
    malicious_rejected_total = 0
    malicious_accepted_total = 0
    final_test_accuracy = None

    if resume_requested:
        if checkpoint is None:
            raise RuntimeError("resume was requested without a loaded checkpoint")
        completed_round = int(checkpoint["completed_round"])
        _reconcile_metrics(metrics_path, completed_round)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        cumulative_num_sq = float(checkpoint["cumulative_num_sq"])
        cumulative_den_sq = float(checkpoint["cumulative_den_sq"])
        malicious_constructed_total = int(
            checkpoint["malicious_constructed_total"]
        )
        malicious_rejected_total = int(checkpoint["malicious_rejected_total"])
        malicious_accepted_total = int(checkpoint["malicious_accepted_total"])
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        final_test_accuracy = checkpoint.get("final_test_accuracy")
        _restore_rng_state(checkpoint.get("rng_state"), torch, np)
        start_round = completed_round + 1
        print(
            f"resume AMM run: seed={seed} gamma={gamma:.2f} mode={mode} "
            f"from_round={start_round}/{cfg.rounds}",
            flush=True,
        )

    metrics_mode = "a" if resume_requested else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not resume_requested:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        for round_idx in range(start_round, cfg.rounds + 1):
            q_vectors = []
            local_losses = []
            for client_id, indices in enumerate(partition.client_indices):
                step_seed = _stable_seed(seed, round_idx, client_id, 101)
                local = train_client(
                    model,
                    train_dataset,
                    indices,
                    cfg.local_steps,
                    cfg.batch_size,
                    cfg.local_lr,
                    cfg.momentum,
                    device,
                    step_seed,
                    cfg.num_workers,
                )
                local_losses.append(local.mean_loss)
                q_seed = _stable_seed(seed, round_idx, client_id, 211)
                q_gen = torch.Generator()
                q_gen.manual_seed(q_seed)
                quantized = stochastic_quantize(local.update, cfg.psi, cfg.amax, q_gen)
                q_vectors.append(quantized.q)

            q_stack = torch.stack(q_vectors, dim=0).to(dtype=torch.int64, device="cpu")
            all_q_sum = q_stack.sum(dim=0)
            if mode == "verification_off":
                accepted_clients = cfg.num_clients
                rejected_clients = 0
                amm_reject_rate = 0.0
                benign_accept_rate = 1.0
                rejected_malicious = 0
                reference_q_sum = all_q_sum
                if protocol is None:
                    recovered_q_sum = all_q_sum + len(malicious_ids) * attack_residual
                    relation_failures = 0
                    aggregate_equality = True
                else:
                    recovered_q_sum = protocol.recover_without_verification(
                        q_updates=q_vectors,
                        malformed_clients=malicious_ids,
                        malformed_direction=attack_direction,
                    )
                    if not hasattr(recovered_q_sum, "to"):
                        recovered_q_sum = torch.tensor(
                            recovered_q_sum, dtype=torch.int64
                        )
                    relation_failures = 0
                    aggregate_equality = bool(
                        torch.equal(
                            recovered_q_sum,
                            all_q_sum + len(malicious_ids) * attack_residual,
                        )
                    )
            elif mode == "vmask":
                if protocol is None:
                    raise RuntimeError(
                        "VMASK mode requires protocol_execution=true; "
                        "the full backend is unavailable"
                    )
                else:
                    protocol_seeds = [
                        _stable_seed(seed, round_idx, client_id, 313)
                        for client_id in range(cfg.num_clients)
                    ]
                    round_result = protocol.execute_round(
                        sid=f"{cfg.experiment_name}:{seed}:{gamma:.4f}:{mode}:{round_idx}",
                        q_updates=q_vectors,
                        seeds=protocol_seeds,
                        malformed_clients=malicious_ids,
                        malformed_direction=attack_direction,
                    )
                    if not round_result.released:
                        raise RuntimeError(
                            "VMASK aggregate was not released: "
                            f"verified={len(round_result.accepted_clients)}, "
                            f"kappa={protocol.cfg.kappa}"
                        )
                    recovered_q_sum = round_result.recovered_q_sum
                    if not hasattr(recovered_q_sum, "to"):
                        recovered_q_sum = torch.tensor(
                            recovered_q_sum, dtype=torch.int64
                        )
                    rejected_set = set(round_result.rejected_clients)
                    rejected_malicious = len(rejected_set.intersection(malicious))
                    rejected_benign = len(rejected_set.difference(malicious))
                    accepted_clients = len(round_result.accepted_clients)
                    rejected_clients = len(round_result.rejected_clients)
                    amm_reject_rate = (
                        rejected_malicious / len(malicious_ids)
                        if malicious_ids
                        else 0.0
                    )
                    benign_accept_rate = (
                        (len(honest_indices) - rejected_benign) / len(honest_indices)
                        if honest_indices
                        else 1.0
                    )
                    reference_q_sum = q_stack[round_result.accepted_clients].sum(dim=0)
                    relation_failures = int(round_result.relation_failures)
                    aggregate_equality = bool(
                        round_result.aggregate_equality
                        and torch.equal(reference_q_sum, recovered_q_sum)
                    )
            else:
                raise ValueError(f"Unsupported AMM mode: {mode}")

            if not aggregate_equality:
                raise AssertionError(
                    f"protocol aggregate mismatch at seed={seed}, gamma={gamma}, "
                    f"mode={mode}, round={round_idx}"
                )

            agg_err, agg_err_num_sq, agg_err_den_sq = aggregate_error(
                recovered_q_sum,
                reference_q_sum,
                eps=1e-12,
            )
            malicious_accepted = len(malicious_ids) - rejected_malicious
            malicious_constructed_total += len(malicious_ids)
            malicious_rejected_total += rejected_malicious
            malicious_accepted_total += malicious_accepted
            cumulative_num_sq += agg_err_num_sq
            cumulative_den_sq += agg_err_den_sq
            update = aggregate_update_from_qsum(
                recovered_q_sum, accepted_clients, cfg.psi, device
            )
            add_update_(model, update, cfg.server_lr)
            ev = evaluate_model(
                model, test_dataset, cfg.eval_batch_size, device, cfg.num_workers
            )

            writer.writerow(
                {
                    "dataset": cfg.dataset,
                    "partition": cfg.partition,
                    "dirichlet_alpha": cfg.dirichlet_alpha,
                    "seed": seed,
                    "gamma": gamma,
                    "malicious_clients": len(malicious_ids),
                    "mode": mode,
                    "round": round_idx,
                    "test_accuracy": ev.accuracy,
                    "test_loss": ev.loss,
                    "mean_local_loss": float(np.mean(local_losses)),
                    "accepted_clients": accepted_clients,
                    "rejected_clients": rejected_clients,
                    "amm_reject_rate": amm_reject_rate,
                    "malicious_submissions_constructed": len(malicious_ids),
                    "malicious_submissions_rejected": rejected_malicious,
                    "malicious_submissions_accepted": malicious_accepted,
                    "benign_accept_rate": benign_accept_rate,
                    "agg_err": agg_err,
                    "agg_err_num_sq": agg_err_num_sq,
                    "agg_err_den_sq": agg_err_den_sq,
                    "max_aggregate_magnitude": int(recovered_q_sum.abs().max().item()),
                    "attack_residual_l2_sq": int(
                        torch.sum(attack_residual * attack_residual).item()
                    ),
                    "attack_limb0_abs": cfg.attack_limb0_abs,
                    "attack_vector_seed": cfg.attack_vector_seed,
                    "protocol_execution": cfg.protocol_execution,
                    "relation_failures": relation_failures,
                    "aggregate_equality": aggregate_equality,
                }
            )


            handle.flush()
            os.fsync(handle.fileno())
            final_test_accuracy = float(ev.accuracy)
            if round_idx % int(cfg.checkpoint_every) == 0 or round_idx == cfg.rounds:
                checkpoint = {
                    "version": 1,
                    "completed_round": round_idx,
                    "run_identity": run_identity,
                    "model_state_dict": _cpu_state_dict(model),
                    "cumulative_num_sq": cumulative_num_sq,
                    "cumulative_den_sq": cumulative_den_sq,
                    "malicious_constructed_total": malicious_constructed_total,
                    "malicious_rejected_total": malicious_rejected_total,
                    "malicious_accepted_total": malicious_accepted_total,
                    "final_test_accuracy": final_test_accuracy,
                    "elapsed_seconds": elapsed_before_resume
                    + (time.perf_counter() - started),
                    "rng_state": _capture_rng_state(torch, np),
                }
                _atomic_torch_save(
                    checkpoint,
                    checkpoint_path,
                    checkpoint_tmp_path,
                    torch,
                )
            print(
                f"seed={seed} gamma={gamma:.2f} mode={mode} round={round_idx}/{cfg.rounds} "
                f"acc={ev.accuracy:.4f} aggerr={agg_err:.4e} "
                f"accepted={accepted_clients} rejected={rejected_clients}",
                flush=True,
            )

    if final_test_accuracy is None:
        raise RuntimeError("completed checkpoint is missing final_test_accuracy")
    elapsed_seconds = elapsed_before_resume + (time.perf_counter() - started)
    cumulative_agg_err = (cumulative_num_sq**0.5) / (cumulative_den_sq**0.5 + 1e-12)
    run_summary = {
        "dataset": cfg.dataset,
        "partition": cfg.partition,
        "alpha": cfg.dirichlet_alpha,
        "gamma": gamma,
        "seed": seed,
        "mode": mode,
        "vmask_backend": cfg.vmask_backend,
        "rounds": cfg.rounds,
        "final_test_accuracy": final_test_accuracy,
        "AggErr_cum": cumulative_agg_err,
        "malicious_submissions_constructed": malicious_constructed_total,
        "malicious_submissions_rejected": malicious_rejected_total,
        "malicious_submissions_accepted": malicious_accepted_total,
        "rejection_rate": (
            malicious_rejected_total / malicious_constructed_total
            if malicious_constructed_total
            else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_write_json(summary_path, run_summary)


def validate_arithmetic_mismatch_attack(cfg: AmmConfig) -> None:
    if int(cfg.attack_limb0_abs) <= int(cfg.amax):
        raise ValueError(
            "attack_limb0_abs must be > amax so that q_i - 2*m_i^(0) "
            "is outside [-amax, amax] for every legal q_i"
        )
    if int(cfg.attack_limb0_abs) > int(cfg.r0):
        raise ValueError("attack_limb0_abs must be within the lowest mask limb range")
    if int(cfg.mask_limbs) < 1:
        raise ValueError("mask_limbs must be positive")
    if bool(cfg.protocol_execution) and int(cfg.attack_limb0_abs) != int(cfg.amax) + 1:
        raise ValueError("protocol execution expects attack_limb0_abs = Amax + 1")


def build_arithmetic_mismatch_residual(cfg: AmmConfig):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(cfg.attack_vector_seed))
    signs = torch.randint(
        low=0,
        high=2,
        size=(int(cfg.expected_dimension),),
        dtype=torch.int64,
        generator=generator,
    )
    signs = signs * 2 - 1
    limb0_abs = int(cfg.attack_limb0_abs)



    return (2 * limb0_abs) * signs


def aggregate_error(recovered_q_sum, reference_q_sum, eps: float):
    import torch

    diff = (recovered_q_sum - reference_q_sum).to(dtype=torch.float64)
    ref = reference_q_sum.to(dtype=torch.float64)
    numerator_sq = float(torch.sum(diff * diff).item())
    denominator_sq = float(torch.sum(ref * ref).item())
    numerator = numerator_sq**0.5
    denominator = denominator_sq**0.5 + float(eps)
    return float(numerator / denominator), numerator_sq, denominator_sq


def _stable_seed(seed: int, round_idx: int, client_id: int, salt: int) -> int:
    return int(seed * 10_000_000 + round_idx * 1_000 + client_id + salt)


def build_protocol(cfg: AmmConfig) -> VMaskProtocol:
    profile = cfg.protocol_profile.lower()
    kwargs = {
        "modulus": int(cfg.protocol_modulus),
        "committee_size": int(cfg.protocol_committee_size),
        "threshold": int(cfg.protocol_threshold),
        "kappa": int(cfg.protocol_kappa),
    }
    if profile == "p80":
        protocol_cfg = profile_p80(**kwargs)
    elif profile == "p100":
        protocol_cfg = profile_p100(**kwargs)
    else:
        raise ValueError(f"Unsupported protocol profile: {cfg.protocol_profile}")
    return VMaskProtocol(
        protocol_cfg, dimension=cfg.expected_dimension, max_clients=cfg.num_clients
    )


def _partition_digest(client_indices) -> str:
    digest = hashlib.sha256()
    for client_id, indices in enumerate(client_indices):
        digest.update(int(client_id).to_bytes(8, "little", signed=False))
        digest.update(len(indices).to_bytes(8, "little", signed=False))
        for index in indices:
            digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _run_identity(run_config: dict[str, object]) -> dict[str, object]:
    payload = dict(run_config)
    for operational_key in ("overwrite", "skip_completed", "checkpoint_every"):
        payload.pop(operational_key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def _cpu_state_dict(model) -> dict[str, object]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _capture_rng_state(torch, np) -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, torch, np) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload, path: Path, tmp_path: Path, torch) -> None:
    torch.save(payload, tmp_path)
    with tmp_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _load_checkpoint(path: Path, torch):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"could not load checkpoint {path}: {exc}") from exc


def _validate_checkpoint(checkpoint, run_identity, rounds: int) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise RuntimeError("unsupported or malformed AMM checkpoint")
    if checkpoint.get("run_identity") != run_identity:
        stored = checkpoint.get("run_identity", {}).get("sha256", "missing")
        current = run_identity["sha256"]
        raise RuntimeError(
            "checkpoint configuration/partition mismatch: "
            f"stored={stored}, current={current}; use --overwrite only if a fresh run is intended"
        )
    completed_round = int(checkpoint.get("completed_round", -1))
    if completed_round < 1 or completed_round > int(rounds):
        raise RuntimeError(
            f"checkpoint completed_round={completed_round} is outside [1, {rounds}]"
        )
    required = {
        "model_state_dict",
        "cumulative_num_sq",
        "cumulative_den_sq",
        "malicious_constructed_total",
        "malicious_rejected_total",
        "malicious_accepted_total",
        "final_test_accuracy",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(f"checkpoint is missing required fields: {missing}")


def _reconcile_metrics(path: Path, completed_round: int) -> None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"could not read metrics for resume: {path}") from exc
    if fieldnames != FIELDS:
        raise RuntimeError(
            f"metrics schema does not match the checkpoint-capable runner: {path}"
        )

    kept_rows = []
    for expected_round, row in enumerate(rows, start=1):
        try:
            row_round = int(row["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid metrics row {expected_round} in {path}") from exc
        if row_round > completed_round:
            break
        if row_round != expected_round:
            raise RuntimeError(
                f"metrics rounds are not contiguous in {path}: "
                f"expected {expected_round}, found {row_round}"
            )
        kept_rows.append(row)

    if len(kept_rows) != completed_round:
        raise RuntimeError(
            f"checkpoint is at round {completed_round}, but metrics contain only "
            f"{len(kept_rows)} durable matching rows"
        )
    if len(kept_rows) != len(rows):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _metrics_complete(path: Path, rounds: int) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, csv.Error):
        return False
    if not rows:
        return False
    try:
        max_round = max(int(row["round"]) for row in rows)
    except (KeyError, TypeError, ValueError):
        return False
    return max_round >= rounds and len(rows) >= rounds


if __name__ == "__main__":
    raise SystemExit(main())
