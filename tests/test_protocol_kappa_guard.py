from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmask_exp.protocol.config import profile_p80
from vmask_exp.protocol.pipeline import VMaskProtocol


class KappaGuardTests(unittest.TestCase):
    dimension = 8

    def setUp(self) -> None:
        self.cfg = profile_p80(
            committee_size=5,
            threshold=3,
            kappa=14,
        )
        self.protocol = VMaskProtocol(
            self.cfg,
            dimension=self.dimension,
            max_clients=20,
        )

    def _inputs(self, count: int) -> tuple[list[list[int]], list[int]]:
        updates = [
            [((client_id * 3 + coordinate) % 11) - 5 for coordinate in range(self.dimension)]
            for client_id in range(count)
        ]
        seeds = [91000 + client_id for client_id in range(count)]
        return updates, seeds

    def _execute(self, count: int):
        updates, seeds = self._inputs(count)
        expected_mask = [0 for _ in range(self.dimension)]
        original_make_submission = self.protocol.make_submission

        def capture_submission(**kwargs):
            submission = original_make_submission(**kwargs)
            for coordinate, value in enumerate(submission.witness_designated_output):
                expected_mask[coordinate] += int(value)
            return submission

        self.protocol.make_submission = capture_submission
        try:
            result = self.protocol.execute_round(
                sid=f"kappa-boundary-{count}",
                q_updates=updates,
                seeds=seeds,
            )
        finally:
            self.protocol.make_submission = original_make_submission
        return result, updates, expected_mask

    def test_release_boundary_and_aggregate_correctness(self) -> None:
        for count in (20, 14):
            with self.subTest(count=count):
                self.setUp()
                result, updates, expected_mask = self._execute(count)
                expected_sum = [
                    sum(update[coordinate] for update in updates)
                    for coordinate in range(self.dimension)
                ]
                recovered = (
                    result.recovered_q_sum.tolist()
                    if hasattr(result.recovered_q_sum, "tolist")
                    else list(result.recovered_q_sum)
                )
                self.assertTrue(result.released)
                self.assertEqual(result.accepted_clients, list(range(count)))
                self.assertEqual(recovered, expected_sum)
                self.assertEqual(result.aggregate_mask, expected_mask)
                self.assertTrue(result.aggregate_equality)
                self.assertGreater(result.timing.aggregate_recovery_seconds, 0.0)
                self.assertIsNone(result.release_reason)

    def test_no_release_below_kappa(self) -> None:
        for count in (13, 2, 1, 0):
            with self.subTest(count=count):
                self.setUp()
                result, _, _ = self._execute(count)
                self.assertFalse(result.released)
                self.assertEqual(result.accepted_clients, list(range(count)))
                self.assertIsNone(result.recovered_q_sum)
                self.assertIsNone(result.aggregate_mask)
                self.assertIsNone(result.aggregate_equality)
                self.assertEqual(result.timing.aggregate_recovery_seconds, 0.0)
                self.assertEqual(result.release_reason, "insufficient_verified_clients")

    def test_two_client_microbenchmark_kappa_boundary(self) -> None:
        updates, seeds = self._inputs(2)
        strict_result = self.protocol.execute_round(
            sid="kappa-14-two-client-abort",
            q_updates=updates,
            seeds=seeds,
        )
        self.assertFalse(strict_result.released)
        self.assertEqual(strict_result.accepted_clients, [0, 1])
        self.assertEqual(strict_result.release_reason, "insufficient_verified_clients")
        self.assertIsNone(strict_result.recovered_q_sum)

        microbenchmark = VMaskProtocol(
            profile_p80(committee_size=5, threshold=3, kappa=2),
            dimension=self.dimension,
            max_clients=20,
        )
        released_result = microbenchmark.execute_round(
            sid="kappa-2-two-client-release",
            q_updates=updates,
            seeds=seeds,
        )
        expected = [
            sum(update[coordinate] for update in updates)
            for coordinate in range(self.dimension)
        ]
        recovered = (
            released_result.recovered_q_sum.tolist()
            if hasattr(released_result.recovered_q_sum, "tolist")
            else list(released_result.recovered_q_sum)
        )
        self.assertTrue(released_result.released)
        self.assertEqual(released_result.accepted_clients, [0, 1])
        self.assertEqual(recovered, expected)
        self.assertTrue(released_result.aggregate_equality)

    def test_kappa_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "kappa"):
            VMaskProtocol(
                profile_p80(kappa=0),
                dimension=self.dimension,
                max_clients=20,
            )
        with self.assertRaisesRegex(ValueError, "kappa"):
            VMaskProtocol(
                profile_p80(kappa=21),
                dimension=self.dimension,
                max_clients=20,
            )


if __name__ == "__main__":
    unittest.main()
