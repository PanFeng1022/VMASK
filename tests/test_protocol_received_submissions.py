from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

import vmask_exp.protocol.pipeline as native
from vmask_exp.protocol.config import profile_p80


def as_list(value):
    if hasattr(value, "detach"):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in value]


class ReceivedSubmissionContractTests(unittest.TestCase):
    def setUp(self):
        self.sid = "native-receive-contract"
        self.cfg = profile_p80(committee_size=5, threshold=3, kappa=2)
        self.protocol = native.VMaskProtocol(
            self.cfg, dimension=2, max_clients=20,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        self.updates = tuple((cid % 7 - 3, 2 - cid % 5) for cid in range(20))
        self.seeds = tuple(650000 + cid for cid in range(20))

    def received_pair(self):
        slots = [None] * 20
        for cid in (0, 1):
            slots[cid] = self.protocol.make_submission(
                sid=self.sid, client_id=cid,
                q_update=self.updates[cid], seed=self.seeds[cid],
            )
        return slots

    def execute_received(self, slots, **kwargs):
        original_check = native.verify_full
        original_recovery = native.reconstruct_shamir_vector
        checks, accepted_bundles, recoveries = [], {}, []

        def check(config, bundle, beacon):
            metrics = original_check(config, bundle, beacon)
            result = metrics.result
            checks.append(dict(client_id=bundle.statement.client_id,
                               accepted=result.accepted,
                               full=len(metrics.verifier_timings) == 3))
            if result.accepted:
                self.assertEqual(tuple(v.committee_id for v in metrics.verifier_timings),
                                 (44, 11, 33))
                self.assertTrue(result.packet_openings_valid)
                self.assertEqual(result.polynomial_identity_residual, 0)
                self.assertEqual(result.constraint_residual, 0)
                accepted_bundles[bundle.statement.client_id] = bundle
            return metrics

        def recovery(*args, **recovery_kwargs):
            self.assertFalse(recovery_kwargs)
            self.assertGreaterEqual(len(accepted_bundles), self.cfg.kappa)
            packet_maps = [{p.committee_id: p for p in bundle.packets}
                           for bundle in accepted_bundles.values()]
            expected = tuple(tuple(sum(packets[h].designated_shamir_share[j]
                                       for packets in packet_maps) % self.cfg.field_modulus
                                   for j in range(2)) for h in (44, 11, 33))
            self.assertEqual(args, (expected, (4, 1, 3), self.cfg.field_modulus))
            recoveries.append(args)
            return original_recovery(*args, **recovery_kwargs)

        with patch.object(self.protocol, "make_submission",
                          side_effect=AssertionError("supplied snapshot must not invoke prover")), \
                patch.object(native, "verify_full", check), \
                patch.object(native, "reconstruct_shamir_vector", recovery):
            result = self.protocol.execute_round(
                sid=self.sid, q_updates=self.updates, seeds=self.seeds,
                received_submissions=slots, **kwargs,
            )
        self.assertEqual(self.protocol.verification_ids, (44, 11, 33))
        self.assertEqual(self.cfg.threshold, 3)
        self.assertEqual(self.cfg.kappa, 2)
        self.assertEqual(result.accepted_clients,
                         [entry["client_id"] for entry in checks if entry["accepted"]])
        if not result.released:
            self.assertIsNone(result.recovered_q_sum)
            self.assertIsNone(result.aggregate_mask)
            self.assertIsNone(result.aggregate_equality)
            self.assertFalse(recoveries)
        return result, checks, recoveries

    def test_received_length_must_preserve_fixed_roster_before_crypto(self):
        for slots in ([None] * 19, [None] * 21):
            with self.subTest(length=len(slots)):
                with self.assertRaisesRegex(ValueError, "fixed client roster"):
                    self.execute_received(slots)

    def test_received_slot_type_is_rejected_before_crypto(self):
        slots = [None] * 20
        slots[0] = object()
        with self.assertRaisesRegex(ValueError, "ClientSubmission or None"):
            self.execute_received(slots)

    def test_all_absent_submissions_abort_without_check_or_recovery(self):
        result, checks, recoveries = self.execute_received([None] * 20)
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "insufficient_verified_clients")
        self.assertEqual(result.accepted_clients, [])
        self.assertEqual(result.rejected_clients, list(range(20)))
        self.assertEqual(result.timing.client_prove_seconds, ())
        self.assertEqual(result.timing.client_check_seconds, ())
        self.assertFalse(checks)
        self.assertFalse(recoveries)

    def test_wrong_session_is_rejected_before_check(self):
        slots = self.received_pair()
        slots[0] = self.protocol.make_submission(
            sid="another-session", client_id=0,
            q_update=self.updates[0], seed=self.seeds[0],
        )
        result, checks, _ = self.execute_received(slots)
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "insufficient_verified_clients")
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual([entry["client_id"] for entry in checks], [1])

    def test_replayed_client_in_another_slot_is_rejected_before_check(self):
        slots = self.received_pair()
        slots[1] = slots[0]
        result, checks, _ = self.execute_received(slots)
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "insufficient_verified_clients")
        self.assertEqual(result.accepted_clients, [0])
        self.assertEqual([entry["client_id"] for entry in checks], [0])

    def test_tampered_selected_commitment_cannot_reach_recovery(self):
        slots = self.received_pair()
        commitments = list(slots[0].evidence.packet_commitments)
        position = self.protocol.committee_ids.index(44)
        value = commitments[position]
        commitments[position] = bytes([value[0] ^ 1]) + value[1:]
        slots[0] = dataclasses.replace(
            slots[0], evidence=dataclasses.replace(
                slots[0].evidence, packet_commitments=tuple(commitments)))
        result, checks, _ = self.execute_received(slots)
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "insufficient_verified_clients")
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual([(e["client_id"], e["accepted"]) for e in checks],
                         [(0, False), (1, True)])

    def test_received_list_slots_are_snapshotted_before_first_challenge(self):
        slots = self.received_pair()
        original_beacon = native.generate_challenge_seed
        calls = []

        def mutate_caller_list_then_generate_real_beacon():
            if not calls:
                slots[1] = None
                slots.append(None)
            calls.append(True)
            return original_beacon()

        with patch.object(native, "generate_challenge_seed",
                          mutate_caller_list_then_generate_real_beacon):
            result, checks, recoveries = self.execute_received(slots)
        self.assertEqual(len(slots), 21)
        self.assertIsNone(slots[1])
        self.assertTrue(result.released)
        self.assertEqual(result.accepted_clients, [0, 1])
        self.assertEqual(len(checks), 2)
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(as_list(result.recovered_q_sum),
                         [sum(self.updates[cid][j] for cid in (0, 1)) for j in range(2)])
        self.assertTrue(result.aggregate_equality)

    def test_selected_unavailable_preserves_fixed_set_and_aborts(self):
        result, checks, recoveries = self.execute_received(
            self.received_pair(), available_committee_ids=(11, 22, 33, 55))
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "verification_committee_unavailable")
        self.assertEqual(result.accepted_clients, [])
        self.assertFalse(checks)
        self.assertFalse(recoveries)


if __name__ == "__main__":
    unittest.main()
