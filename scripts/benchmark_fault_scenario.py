from __future__ import annotations

import argparse
from contextlib import ExitStack
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
import flint
import psutil
import vmask_exp.protocol.pipeline as pipeline
import vmask_exp.protocol.threshold_snip as snip
from vmask_exp.protocol.config import profile_p80

IDS = (11, 22, 33, 44, 55)
SELECTED = (44, 11, 33)
CASES = ("A3_before", "A3_late6", "A3_late7", "A3_reorder",
         "B1_nonselected", "B2_precheck", "B3_residual", "B4_recovery",
         "C_absent6", "C_absent7")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sources():
    files = sorted((REPO / "src/vmask_exp/protocol").glob("*.py"))
    return {p.relative_to(REPO).as_posix(): sha(p) for p in files}


def as_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(v) for v in value]


class Receiver:

    def __init__(self, case, submissions):
        self.case = case
        self.submissions = submissions
        self.deadline = .200
        self.events = []
        self.received = set()
        self.closed = False
        self.deliveries = []
        self.late_clients = set(range(14, 20)) if case == "A3_late6" else (
            set(range(13, 20)) if case == "A3_late7" else set())
        for cid in range(20):
            for h in IDS:
                if cid in self.late_clients and h == SELECTED[0]:
                    delay = .350
                elif case == "A3_reorder":
                    delay = .020 + .005 * (19 - cid) + .001 * IDS.index(h)
                else:
                    delay = .020 + .002 * cid + .001 * IDS.index(h)
                self.deliveries.append((delay, cid, h))
        self.deliveries.sort()
        self.origin = time.perf_counter()

    def pump(self, through):
        while self.deliveries and self.deliveries[0][0] <= through:
            delay, cid, h = self.deliveries.pop(0)
            remaining = self.origin + delay - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            self.deliver(cid, h, delay)

    def deliver(self, cid, h, delay=None, kind="scheduled_delivery"):
        observed = time.perf_counter() - self.origin
        ignored = self.closed or observed > self.deadline
        if not ignored:
            self.received.add((cid, h))
        self.events.append(dict(client_id=cid, committee_id=h, event=kind,
                                scheduled_delay_seconds=delay,
                                observed_since_origin_seconds=observed,
                                ignored_after_close=ignored))

    def freeze(self):
        self.pump(self.deadline)
        remaining = self.origin + self.deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        self.closed = True
        frozen = {}
        for cid, submission in self.submissions.items():
            packets = tuple(p for p in submission.evidence.verifier_packets
                            if (cid, p.committee_id) in self.received)
            evidence = dataclasses.replace(submission.evidence, verifier_packets=packets)
            frozen[cid] = dataclasses.replace(submission, evidence=evidence)
            assert frozen[cid].gamma is submission.gamma
            assert evidence.packet_commitments is submission.evidence.packet_commitments
            assert all(any(p is original for original in submission.evidence.verifier_packets) for p in packets)
        self.frozen = frozen
        return frozen


def run(case, repetition, dimension=100, clients=20):
    assert case in CASES
    cfg = profile_p80(committee_size=5, threshold=3, kappa=14)
    protocol = pipeline.VMaskProtocol(cfg, dimension, 20, committee_ids=IDS, verification_ids=SELECTED)
    assert protocol.verification_ids == SELECTED
    vectors = tuple(tuple(((j * (cid + 1) + cid + repetition) % 11) - 5 for j in range(dimension))
                    for cid in range(clients))
    seeds = tuple(3100000 + 1000 * repetition + cid for cid in range(clients))
    sid = f"final-fault-20261031-{case}-{repetition}"
    events, made, checks, raw_shares, recoveries = [], [], [], {}, []
    preformed, receiver, received_snapshot, current = {}, None, None, {"client_id": None}
    original_make = protocol.make_submission
    original_check = pipeline.verify_full
    original_recovery = pipeline.reconstruct_shamir_vector
    original_residual = snip.reference.reconstruct_shamir_vector
    original_beacon = pipeline.generate_challenge_seed

    def event(kind, **kwargs):
        row = dict(event=kind, since_start_seconds=time.perf_counter()-started, **kwargs)
        events.append(row)
        print(json.dumps(row), flush=True)

    def genuine_make(**kwargs):
        submission = original_make(**kwargs)
        made.append(kwargs["client_id"])
        event("genuine_prove_complete", client_id=kwargs["client_id"],
              commitments=[c.hex() for c in submission.evidence.packet_commitments])
        return submission

    def supplied(**kwargs):
        cid = kwargs["client_id"]
        assert kwargs["sid"] == sid and kwargs["seed"] == seeds[cid]
        return preformed[cid]

    def check(config, bundle, beacon):
        cid = bundle.statement.client_id
        current["client_id"] = cid
        assert bundle.statement.verification_ids == SELECTED
        assert tuple(p.committee_id for p in bundle.packets if p.committee_id in SELECTED) == (
            tuple(h for h in IDS if h in SELECTED)) if len(bundle.packets) == 5 else True
        event("native_check_enter", client_id=cid, beacon_hex=beacon.hex(), packet_count=len(bundle.packets))
        if case == "B2_precheck" and cid == 0:
            event("committee_fault", phase="before_SNIP_Check", committee_id=SELECTED[0], client_id=cid)
            raise pipeline.CommitteeUnavailableError(SELECTED[0])
        metrics = original_check(config, bundle, beacon)
        assert len(metrics.verifier_timings) in (0, 3)
        full = tuple(v.committee_id for v in metrics.verifier_timings) == SELECTED
        if metrics.result.accepted:
            assert full and metrics.result.packet_openings_valid
            raw_shares[cid] = {p.committee_id: p.designated_shamir_share for p in bundle.packets if p.committee_id in SELECTED}
        checks.append(dict(client_id=cid, accepted=metrics.result.accepted,
                           full_arithmetic_check=full, verifier_ids=[v.committee_id for v in metrics.verifier_timings],
                           packet_openings_valid=metrics.result.packet_openings_valid,
                           polynomial_residual=metrics.result.polynomial_identity_residual,
                           constraint_residual=metrics.result.constraint_residual,
                           total_seconds=metrics.total_seconds))
        event("native_check_complete", **checks[-1])
        current["client_id"] = None
        return metrics

    def residual(*args, **kwargs):
        if case == "B3_residual" and current["client_id"] == 0:
            assert len(args[0]) == 3 and all(len(share) == 2 for share in args[0])
            assert args[1] == (4, 1, 3)
            event("committee_fault", phase="two_residual_reconstruction_after_DE_and_phase_two",
                  committee_id=SELECTED[0], client_id=0, share_points=args[1],
                  residual_input_share_count=len(args[0]))
            raise pipeline.CommitteeUnavailableError(SELECTED[0])
        return original_residual(*args, **kwargs)

    def recovery(*args, **kwargs):
        recoveries.append(args)
        accepted_so_far = [c["client_id"] for c in checks if c["accepted"]]
        expected_shares = tuple(tuple(sum(raw_shares[cid][h][j] for cid in accepted_so_far) % cfg.field_modulus
                                     for j in range(dimension)) for h in SELECTED)
        assert not kwargs and args[0] == expected_shares and args[1] == (4, 1, 3)
        assert args[2] == cfg.field_modulus and len(accepted_so_far) >= cfg.kappa
        event("aggregate_recovery_enter", share_points=args[1], accepted_at_recovery=[c["client_id"] for c in checks if c["accepted"]])
        if case == "B4_recovery":
            event("committee_fault", phase="after_verification_before_aggregate_reconstruction", committee_id=SELECTED[0])
            raise pipeline.CommitteeUnavailableError(SELECTED[0])
        return original_recovery(*args, **kwargs)

    def beacon():

        result = original_beacon()
        event("post_commitment_beacon", beacon_hex=result.hex())
        return result

    started = time.perf_counter()
    proof_preparation_seconds = delivery_wait_seconds = 0.0
    with ExitStack() as stack:
        if case.startswith("A3_"):


            before = time.perf_counter()
            preformed = {cid: genuine_make(sid=sid, client_id=cid, q_update=vectors[cid], seed=seeds[cid])
                         for cid in range(clients)}
            proof_preparation_seconds = time.perf_counter() - before
            receiver = Receiver(case, preformed)
            before = time.perf_counter()
            preformed = receiver.freeze()



            receiver.pump(float("inf"))
            delivery_wait_seconds = time.perf_counter() - before
            event("receiver_closed", deadline_seconds=receiver.deadline, late_clients=sorted(receiver.late_clients))
            stack.enter_context(patch.object(protocol, "make_submission", supplied))
        elif case.startswith("C_absent"):



            valid_count = 14 if case == "C_absent6" else 13
            before = time.perf_counter()
            slots = [None] * clients
            for cid in range(valid_count):
                slots[cid] = genuine_make(
                    sid=sid, client_id=cid, q_update=vectors[cid], seed=seeds[cid]
                )
            proof_preparation_seconds = time.perf_counter() - before
            received_snapshot = tuple(slots)
            event("closed_public_submission_snapshot",
                  received_clients=list(range(valid_count)),
                  absent_clients=list(range(valid_count, clients)))
            stack.enter_context(patch.object(
                protocol, "make_submission",
                side_effect=AssertionError("closed received snapshot must not invoke prover")))
        else:
            stack.enter_context(patch.object(protocol, "make_submission", genuine_make))
        stack.enter_context(patch.object(pipeline, "verify_full", check))
        stack.enter_context(patch.object(pipeline, "reconstruct_shamir_vector", recovery))
        stack.enter_context(patch.object(snip.reference, "reconstruct_shamir_vector", residual))
        stack.enter_context(patch.object(pipeline, "generate_challenge_seed", beacon))
        execute_started = time.perf_counter()
        result = protocol.execute_round(
            sid=sid, q_updates=vectors, seeds=seeds,
            available_committee_ids=(11, 33, 44, 55) if case == "B1_nonselected" else None,
            received_submissions=received_snapshot,
        )
        execute_caller_seconds = time.perf_counter()-execute_started
    terminal_seconds = time.perf_counter()-started
    terminal = dict(released=result.released, accepted=list(result.accepted_clients),
                    rejected=list(result.rejected_clients), output=as_list(result.recovered_q_sum),
                    mask=None if result.aggregate_mask is None else list(result.aggregate_mask))
    events_before_late = len(events)
    if receiver:
        frozen_received = set(receiver.received)
        for cid in sorted(receiver.late_clients):
            receiver.deliver(cid, SELECTED[0], kind="post_terminal_retransmission")
        assert receiver.received == frozen_received and receiver.closed


        assert terminal == dict(released=result.released, accepted=list(result.accepted_clients),
                                rejected=list(result.rejected_clients), output=as_list(result.recovered_q_sum),
                                mask=None if result.aggregate_mask is None else list(result.aggregate_mask))
        assert len(events) == events_before_late
        assert all(e["ignored_after_close"] for e in receiver.events if e["client_id"] in receiver.late_clients and e["committee_id"] == SELECTED[0])
        assert all(e["observed_since_origin_seconds"] <= receiver.deadline
                   for e in receiver.events if not e["ignored_after_close"])
    accepted = [c["client_id"] for c in checks if c["accepted"]]
    assert result.accepted_clients == accepted
    assert all(c["full_arithmetic_check"] and c["packet_openings_valid"] for c in checks if c["accepted"])
    assert protocol.verification_ids == SELECTED and cfg.threshold == 3 and cfg.kappa == 14
    expected = [sum(vectors[cid][j] for cid in accepted) for j in range(dimension)]
    if result.released:
        assert len(accepted) >= cfg.kappa and len(recoveries) == 1
        args = recoveries[0]
        expected_shares = tuple(tuple(sum(raw_shares[cid][h][j] for cid in accepted) % cfg.field_modulus
                                     for j in range(dimension)) for h in SELECTED)
        assert args[0] == expected_shares and args[1] == (4, 1, 3)
        assert [v % cfg.modulus for v in terminal["output"]] == [v % cfg.modulus for v in expected]
        assert result.aggregate_equality
    else:
        assert result.recovered_q_sum is None and result.aggregate_mask is None
        assert len(recoveries) == (1 if case == "B4_recovery" else 0)
    if case.startswith("A3_"):
        valid = clients - len(receiver.late_clients)
        assert accepted == list(range(valid)) and result.released == (valid >= cfg.kappa)
        assert len(made) == clients and len(checks) == clients
        assert sum(c["full_arithmetic_check"] for c in checks) == valid
        assert result.rejected_clients == sorted(receiver.late_clients)
        if not result.released:
            assert result.release_reason == "insufficient_verified_clients"
    elif case.startswith("C_absent"):
        valid = 14 if case == "C_absent6" else 13
        assert accepted == list(range(valid))
        assert result.rejected_clients == list(range(valid, clients))
        assert len(made) == valid and len(checks) == valid
        assert result.released == (valid >= cfg.kappa)
        if not result.released:
            assert result.release_reason == "insufficient_verified_clients"
    elif case == "B1_nonselected":
        assert result.released and len(accepted) == clients
    else:
        assert not result.released and result.release_reason == "verification_committee_unavailable"
        assert len(accepted) == (clients if case == "B4_recovery" else 0)
    invariant_status = {
        "I1_accepted_full_native_check": True,
        "I2_same_fixed_T_and_verified_original_shares": True,
        "I3_aggregate_correctness": "PASS" if result.released else "NOT_APPLICABLE_NO_RELEASE",
        "I4_release_threshold": True,
        "I5_selected_failure_safe_abort": True if case.startswith("B") and case != "B1_nonselected" else "NOT_APPLICABLE",
        "I6_no_unverified_recovery": True,
        "late_packets_cannot_reopen_terminal_round": True if receiver else "NOT_APPLICABLE",
    }
    return dict(case_id=case, repetition=repetition, sid=sid, status="complete",
                configuration=dict(profile="P80", d=dimension, Nt=clients, Nmax=20, n=5, tau=3, kappa=14,
                                   committee_ids=IDS, verification_ids=SELECTED),
                inputs=vectors, seeds=seeds, input_sha256=digest(vectors),
                released=result.released, release_reason=result.release_reason,
                accepted_clients=accepted, rejected_clients=result.rejected_clients,
                full_arithmetic_checks=sum(c["full_arithmetic_check"] for c in checks),
                check_attempts=sum(e["event"]=="native_check_enter" for e in events),
                genuine_prove_calls=len(made), actual_prove_client_ids=made,
                checks=checks, events=events, result_output=terminal["output"], expected_sum=expected,
                invariants=invariant_status, recovery_attempts=len(recoveries),
                timings=dict(proof_preparation_seconds=proof_preparation_seconds,
                             controlled_delivery_wait_seconds=delivery_wait_seconds,
                             native_execute_caller_seconds=execute_caller_seconds,
                             total_local_workflow_to_terminal_seconds=terminal_seconds,
                             native_reported=dataclasses.asdict(result.timing),
                             sum_full_native_prove_seconds=sum(result.timing.client_prove_seconds)),
                transport=None if not receiver else dict(deadline_seconds=receiver.deadline,
                    deadline_owner="independent_harness_not_original_protocol", origin="after_all_real_proofs_and_commitments",
                    events=receiver.events, frozen_received_count=len(receiver.received),
                    late_clients=sorted(receiver.late_clients), terminal_round_reopened=False),
                reception_snapshot=None if received_snapshot is None else dict(
                    fixed_roster_size=len(received_snapshot),
                    received_clients=[i for i,item in enumerate(received_snapshot) if item is not None],
                    absent_clients=[i for i,item in enumerate(received_snapshot) if item is None]),
                real_network=False, automatic_failure_detection=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output=args.output.resolve();args.output.parent.mkdir(parents=True,exist_ok=True)
    assert not args.output.exists(), "Refusing to overwrite results"
    assert hasattr(flint, "fmpz_mod_poly_ctx")
    before = sources()
    envelope = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_hashes_before=before,
                    worker_sha256=sha(__file__), backend=snip.BACKEND_NAME,
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                        flint=flint.__version__, psutil=psutil.__version__,
                        available_memory_bytes=psutil.virtual_memory().available))
    try:
        setup_start = time.perf_counter()
        info = snip.prepare_public_interpolation(
            profile_p80().field_modulus, 231 * args.dimension)
        envelope["public_setup_seconds"] = time.perf_counter()-setup_start
        envelope["public_setup_point_count"] = info.point_count
        envelope["record"] = run(args.case, args.repetition, dimension=args.dimension)
        envelope["status"] = "complete"
    except BaseException as error:
        envelope.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        envelope["source_hashes_after"] = sources()
        assert before == envelope["source_hashes_after"]
        envelope["process_memory"] = dict(psutil.Process().memory_info()._asdict())
        save(args.output, envelope)


if __name__ == "__main__":
    main()
