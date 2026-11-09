from __future__ import annotations

import unittest
from unittest.mock import patch

import vmask_exp.protocol.pipeline as pipeline
from vmask_exp.protocol.config import profile_p80
from vmask_exp.protocol.pipeline import CommitteeUnavailableError, VMaskProtocol


def as_list(values):
    if hasattr(values, "detach"):
        return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in values]


class CommitteeUnavailableRegressionTests(unittest.TestCase):


    def setUp(self):
        self.cfg = profile_p80(committee_size=5, threshold=3, kappa=2)
        self.protocol = VMaskProtocol(
            self.cfg, dimension=10, max_clients=20,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        self.selected = self.protocol.verification_ids
        self.updates = (tuple(i % 5 - 2 for i in range(10)),
                        tuple(3 - i % 7 for i in range(10)))
        self.seeds = (831001, 831002)

    def execute(self, **kwargs):
        return self.protocol.execute_round(
            sid="committee-unavailability-regression",
            q_updates=self.updates, seeds=self.seeds, **kwargs,
        )

    def assert_abort(self, result):
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "verification_committee_unavailable")
        self.assertIsNone(result.recovered_q_sum)
        self.assertIsNone(result.aggregate_mask)
        self.assertIsNone(result.aggregate_equality)
        self.assertEqual(result.timing.aggregate_recovery_seconds, 0.0)
        self.assertEqual(self.protocol.verification_ids, self.selected)
        self.assertEqual(self.protocol.cfg.threshold, 3)
        self.assertEqual(self.protocol.cfg.kappa, 2)

    def test_selected_member_unavailable_during_check_returns_abort(self):
        original = pipeline.verify_full

        def fail_second_check(cfg, bundle, beacon):
            if bundle.statement.client_id == 1:
                raise CommitteeUnavailableError(self.selected[0])
            return original(cfg, bundle, beacon)

        with patch.object(pipeline, "verify_full", fail_second_check), \
                patch.object(pipeline, "reconstruct_shamir_vector") as reconstruction:
            result = self.execute()
        self.assert_abort(result)
        self.assertEqual(result.accepted_clients, [0])
        self.assertEqual(result.rejected_clients, [])
        self.assertEqual(len(result.timing.client_prove_seconds), 2)
        self.assertEqual(len(result.timing.client_check_seconds), 1)
        reconstruction.assert_not_called()

    def test_selected_member_unavailable_at_recovery_returns_abort(self):
        with patch.object(pipeline, "reconstruct_shamir_vector",
                          side_effect=CommitteeUnavailableError(self.selected[-1])) as reconstruction:
            result = self.execute()
        self.assert_abort(result)
        self.assertEqual(result.accepted_clients, [0, 1])
        self.assertEqual(result.rejected_clients, [])
        self.assertEqual(len(result.timing.client_check_seconds), 2)
        reconstruction.assert_called_once()

    def test_plain_errors_during_check_are_not_swallowed(self):
        for error_type in (ConnectionError, ValueError, RuntimeError):
            with self.subTest(error=error_type.__name__):
                error = error_type("unrelated check failure")
                with patch.object(pipeline, "verify_full", side_effect=error):
                    with self.assertRaises(error_type) as caught:
                        self.execute()
                self.assertIs(caught.exception, error)

    def test_plain_errors_at_recovery_are_not_swallowed(self):
        for error_type in (ConnectionError, ValueError, RuntimeError):
            with self.subTest(error=error_type.__name__):
                error = error_type("unrelated recovery failure")
                with patch.object(pipeline, "reconstruct_shamir_vector", side_effect=error):
                    with self.assertRaises(error_type) as caught:
                        self.execute()
                self.assertIs(caught.exception, error)

    def test_nonselected_typed_errors_do_not_bypass_checks(self):
        for target in ("verify_full", "reconstruct_shamir_vector"):
            with self.subTest(target=target):
                error = CommitteeUnavailableError(22)
                with patch.object(pipeline, target, side_effect=error):
                    with self.assertRaises(CommitteeUnavailableError) as caught:
                        self.execute()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.protocol.verification_ids, self.selected)

    def test_nonselected_member_unavailable_at_entry_does_not_abort(self):
        result = self.execute(available_committee_ids=(11, 33, 44, 55))
        self.assertTrue(result.released)
        self.assertTrue(result.aggregate_equality)
        self.assertEqual(result.accepted_clients, [0, 1])
        self.assertEqual(as_list(result.recovered_q_sum),
                         [sum(values) for values in zip(*self.updates)])
        self.assertEqual(self.protocol.verification_ids, self.selected)

    def test_selected_member_unavailable_at_entry_keeps_existing_abort(self):
        with patch.object(self.protocol, "make_submission") as prove:
            result = self.execute(available_committee_ids=(11, 22, 33, 55))
        self.assert_abort(result)
        self.assertEqual(result.accepted_clients, [])
        prove.assert_not_called()

    def test_release_threshold_is_not_reduced(self):
        cfg = profile_p80(committee_size=5, threshold=3, kappa=3)
        protocol = VMaskProtocol(cfg, dimension=10, max_clients=20,
                                committee_ids=(11, 22, 33, 44, 55),
                                verification_ids=self.selected)
        result = protocol.execute_round(sid="kappa-not-reduced", q_updates=self.updates,
                                        seeds=self.seeds)
        self.assertFalse(result.released)
        self.assertEqual(result.release_reason, "insufficient_verified_clients")
        self.assertEqual(result.accepted_clients, [0, 1])
        self.assertIsNone(result.recovered_q_sum)
        self.assertIsNone(result.aggregate_mask)
        self.assertEqual(protocol.cfg.kappa, 3)


if __name__ == "__main__":
    unittest.main()
