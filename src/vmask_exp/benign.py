from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, config_dict, set_global_seed
from .data import dataset_labels, load_dataset, partition_dataset, save_label_distribution
from .model import assert_parameter_count, create_model
from .params import add_update_, clone_model
from .protocol import VMaskProtocol
from .protocol.config import profile_p100, profile_p80
from .quantization import (
    aggregate_update_from_qsum,
    check_modulus,
    relative_eq_error,
    signed_mod_decode,
    stochastic_quantize,
)
from .trainer import evaluate_model, train_client


METRIC_FIELDS = [
    "seed",
    "partition",
    "round",
    "method",
    "test_accuracy",
    "test_loss",
    "mean_local_loss",
    "adjusted_coordinates",
    "total_coordinates",
    "adjustment_rate",
    "maximum_aggregate_magnitude",
    "aggregate_equality",
    "aggregate_eq_error",
    "modulus",
    "modulus_bound_ok",
    "wraparound_ok",
    "protocol_execution",
    "verified_clients",
    "rejected_clients",
    "relation_failures",
]


def run_benign_experiment(cfg: ExperimentConfig, device) -> None:
    check_modulus(cfg.modulus, cfg.participating_clients, cfg.amax)
    train_dataset, test_dataset = load_dataset(cfg.dataset, cfg.data_dir, cfg.download)
    labels = dataset_labels(train_dataset)

    for seed in cfg.seeds:
        for partition in cfg.partitions:
            run_seed_partition(cfg, seed, partition, train_dataset, test_dataset, labels, device)


def run_seed_partition(
    cfg: ExperimentConfig,
    seed: int,
    partition: str,
    train_dataset,
    test_dataset,
    labels: np.ndarray,
    device,
) -> None:
    import torch

    if cfg.participating_clients != cfg.num_clients:
        raise ValueError("This benign experiment expects full client participation each round")

    out_dir = Path(cfg.results_dir) / f"seed_{seed}" / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if cfg.overwrite:
        for stale in [metrics_path, out_dir / "config.json", out_dir / "label_distribution.csv"]:
            if stale.exists():
                stale.unlink()
    elif metrics_path.exists():
        if cfg.skip_completed and _metrics_complete(metrics_path, cfg.rounds, expected_rows_per_round=2):
            print(f"skip completed benign run: seed={seed} partition={partition} rounds={cfg.rounds}")
            return
        if cfg.skip_completed:
            print(f"restart incomplete benign run: seed={seed} partition={partition} rounds={cfg.rounds}")
            for stale in [metrics_path, out_dir / "config.json", out_dir / "label_distribution.csv"]:
                if stale.exists():
                    stale.unlink()
        else:
            raise FileExistsError(
                f"{metrics_path} already exists but is not marked for overwrite. "
                "Use --overwrite to replace it, or remove the incomplete run directory."
            )

    (out_dir / "config.json").write_text(
        json.dumps(config_dict(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    partition_seed = _stable_seed(seed, partition, 0, 0, salt=11)
    partition_result = partition_dataset(
        labels=labels,
        num_clients=cfg.num_clients,
        partition=partition,
        seed=partition_seed,
        alpha=cfg.dirichlet_alpha,
    )
    save_label_distribution(out_dir / "label_distribution.csv", partition_result.label_counts)

    set_global_seed(seed)
    base_model = create_model(cfg.model).to(device)
    assert_parameter_count(base_model, cfg.expected_dimension)
    fedavg_model = clone_model(base_model).to(device)
    q_model = clone_model(base_model).to(device)
    protocol = _build_protocol(cfg) if cfg.protocol_execution else None

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for round_idx in range(1, cfg.rounds + 1):
            fed_updates = []
            fed_losses = []
            q_vectors = []
            q_losses = []
            adjusted_coordinates = 0
            total_coordinates = 0

            for client_id, indices in enumerate(partition_result.client_indices):
                step_seed = _stable_seed(seed, partition, round_idx, client_id, salt=101)

                fed_result = train_client(
                    fedavg_model,
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
                fed_updates.append(fed_result.update)
                fed_losses.append(fed_result.mean_loss)

                q_result = train_client(
                    q_model,
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
                q_losses.append(q_result.mean_loss)

                q_seed = _stable_seed(seed, partition, round_idx, client_id, salt=211)
                generator = torch.Generator()
                generator.manual_seed(q_seed)
                quantized = stochastic_quantize(q_result.update, cfg.psi, cfg.amax, generator)
                q_vectors.append(quantized.q)
                adjusted_coordinates += quantized.adjusted_coordinates
                total_coordinates += quantized.total_coordinates

            fed_mean_update = torch.stack(fed_updates, dim=0).mean(dim=0)
            add_update_(fedavg_model, fed_mean_update, cfg.server_lr)

            q_sum = torch.stack(q_vectors, dim=0).sum(dim=0)
            signed_q_sum = signed_mod_decode(q_sum, cfg.modulus)
            wraparound_ok = bool(torch.equal(q_sum, signed_q_sum))
            max_aggregate = int(q_sum.abs().max().item())
            modulus_bound_ok = bool(max_aggregate < cfg.modulus / 2)
            if not wraparound_ok or not modulus_bound_ok:
                raise RuntimeError(
                    f"Modular wrap-around detected at seed={seed}, partition={partition}, "
                    f"round={round_idx}, max={max_aggregate}, M={cfg.modulus}"
                )

            reference_quantized_aggregate = q_sum
            if protocol is None:
                vmask_aggregate = signed_q_sum
                verified_clients = cfg.num_clients
                rejected_clients = 0
                relation_failures = 0
                aggregate_equality = bool(torch.equal(reference_quantized_aggregate, vmask_aggregate))
            else:
                protocol_seeds = [
                    _stable_seed(seed, partition, round_idx, client_id, salt=313)
                    for client_id in range(cfg.num_clients)
                ]
                round_result = protocol.execute_round(
                    sid=f"{cfg.experiment_name}:{seed}:{partition}:{round_idx}",
                    q_updates=q_vectors,
                    seeds=protocol_seeds,
                )
                vmask_aggregate = round_result.recovered_q_sum
                if not hasattr(vmask_aggregate, "to"):
                    vmask_aggregate = torch.tensor(vmask_aggregate, dtype=torch.int64)
                verified_clients = len(round_result.accepted_clients)
                rejected_clients = len(round_result.rejected_clients)
                relation_failures = int(round_result.relation_failures)
                aggregate_equality = bool(
                    round_result.aggregate_equality
                    and torch.equal(reference_quantized_aggregate, vmask_aggregate)
                )
            if not aggregate_equality:
                raise AssertionError("VMASK recovered aggregate diverged from the direct quantized aggregate")
            eq_error = relative_eq_error(reference_quantized_aggregate, vmask_aggregate)

            decoded_mean_update = aggregate_update_from_qsum(vmask_aggregate, verified_clients, cfg.psi, device)
            add_update_(q_model, decoded_mean_update, cfg.server_lr)

            fed_eval = evaluate_model(
                fedavg_model,
                test_dataset,
                cfg.eval_batch_size,
                device,
                cfg.num_workers,
            )
            q_eval = evaluate_model(
                q_model,
                test_dataset,
                cfg.eval_batch_size,
                device,
                cfg.num_workers,
            )

            adjustment_rate = adjusted_coordinates / total_coordinates
            rows = [
                {
                    "seed": seed,
                    "partition": partition,
                    "round": round_idx,
                    "method": "FedAvg",
                    "test_accuracy": fed_eval.accuracy,
                    "test_loss": fed_eval.loss,
                    "mean_local_loss": float(np.mean(fed_losses)),
                    "adjusted_coordinates": "",
                    "total_coordinates": "",
                    "adjustment_rate": "",
                    "maximum_aggregate_magnitude": "",
                    "aggregate_equality": "",
                    "aggregate_eq_error": "",
                    "modulus": cfg.modulus,
                    "modulus_bound_ok": modulus_bound_ok,
                    "wraparound_ok": wraparound_ok,
                    "protocol_execution": cfg.protocol_execution,
                    "verified_clients": "",
                    "rejected_clients": "",
                    "relation_failures": "",
                },
                {
                    "seed": seed,
                    "partition": partition,
                    "round": round_idx,
                    "method": "VMASK",
                    "test_accuracy": q_eval.accuracy,
                    "test_loss": q_eval.loss,
                    "mean_local_loss": float(np.mean(q_losses)),
                    "adjusted_coordinates": adjusted_coordinates,
                    "total_coordinates": total_coordinates,
                    "adjustment_rate": adjustment_rate,
                    "maximum_aggregate_magnitude": max_aggregate,
                    "aggregate_equality": aggregate_equality,
                    "aggregate_eq_error": eq_error,
                    "modulus": cfg.modulus,
                    "modulus_bound_ok": modulus_bound_ok,
                    "wraparound_ok": wraparound_ok,
                    "protocol_execution": cfg.protocol_execution,
                    "verified_clients": verified_clients,
                    "rejected_clients": rejected_clients,
                    "relation_failures": relation_failures,
                },
            ]
            writer.writerows(rows)
            handle.flush()

            print(
                f"seed={seed} partition={partition} round={round_idx}/{cfg.rounds} "
                f"fedavg_acc={fed_eval.accuracy:.4f} vmask_acc={q_eval.accuracy:.4f} "
                f"adj={adjustment_rate:.6f} maxagg={max_aggregate}",
                flush=True,
            )


def _stable_seed(seed: int, partition: str, round_idx: int, client_id: int, salt: int) -> int:
    partition_offset = 0 if partition == "iid" else 100_000
    return int(seed * 10_000_000 + partition_offset + round_idx * 1_000 + client_id + salt)


def _build_protocol(cfg: ExperimentConfig) -> VMaskProtocol:
    profile = cfg.protocol_profile.lower()
    kwargs = {
        "modulus": int(cfg.protocol_modulus),
        "committee_size": int(cfg.protocol_committee_size),
        "threshold": int(cfg.protocol_threshold),
    }
    if profile == "p80":
        protocol_cfg = profile_p80(**kwargs)
    elif profile == "p100":
        protocol_cfg = profile_p100(**kwargs)
    else:
        raise ValueError(f"Unsupported protocol profile: {cfg.protocol_profile}")
    return VMaskProtocol(protocol_cfg, dimension=cfg.expected_dimension, max_clients=cfg.num_clients)


def _metrics_complete(path: Path, rounds: int, expected_rows_per_round: int) -> bool:
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
    return max_round >= rounds and len(rows) >= rounds * expected_rows_per_round
