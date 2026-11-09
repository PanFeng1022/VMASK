from __future__ import annotations

import unittest
from unittest import mock

from vmask_exp.protocol import threshold_snip
from vmask_exp.protocol.config import profile_p80
from vmask_exp.protocol.pipeline import VMaskProtocol


class ThresholdSnipPublicPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        threshold_snip.clear_public_interpolation_cache()
        self.cfg = profile_p80(committee_size=5, threshold=3, kappa=1)

    def tearDown(self) -> None:
        threshold_snip.clear_public_interpolation_cache()

    def test_same_public_key_is_built_once(self) -> None:
        with mock.patch.object(
            threshold_snip,
            "ConsecutiveInterpolator",
            wraps=threshold_snip.ConsecutiveInterpolator,
        ) as constructor:
            first = threshold_snip.prepare_public_interpolation(
                self.cfg.field_modulus,
                7,
            )
            second = threshold_snip.prepare_public_interpolation(
                self.cfg.field_modulus,
                7,
            )
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(
            threshold_snip.public_interpolation_cache_info(),
            (first,),
        )

    def test_different_public_key_replaces_cache_entry(self) -> None:
        first = threshold_snip.prepare_public_interpolation(
            self.cfg.field_modulus,
            7,
        )
        second = threshold_snip.prepare_public_interpolation(
            self.cfg.field_modulus,
            8,
        )
        third = threshold_snip.prepare_public_interpolation(
            65537,
            8,
        )
        self.assertNotEqual(first.multiplication_count, second.multiplication_count)
        self.assertNotEqual(second.field_modulus, third.field_modulus)
        self.assertEqual(
            threshold_snip.public_interpolation_cache_info(),
            (third,),
        )

    def test_cache_does_not_change_deterministic_proof_material(self) -> None:
        protocol = VMaskProtocol(
            self.cfg,
            dimension=2,
            max_clients=4,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        arguments = {
            "sid": "cache-equivalence",
            "client_id": 0,
            "q_update": (1, -2),
            "seed": 91337,
        }


        with mock.patch.object(threshold_snip.secrets, "token_bytes", return_value=b"c" * 32):
            cold = protocol.make_submission(**arguments)
            warm = protocol.make_submission(**arguments)
        self.assertEqual(cold.gamma, warm.gamma)
        self.assertEqual(cold.evidence, warm.evidence)
        self.assertEqual(
            cold.witness_designated_output,
            warm.witness_designated_output,
        )

    def test_lagrange_weights_are_computed_once_per_verification(self) -> None:
        protocol = VMaskProtocol(
            self.cfg,
            dimension=2,
            max_clients=4,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        submission = protocol.make_submission(
            sid="weight-reuse",
            client_id=0,
            q_update=(1, -2),
            seed=91401,
        )
        bundle = protocol._formal_bundle(submission)
        with mock.patch.object(
            threshold_snip,
            "consecutive_lagrange_weights",
            wraps=threshold_snip.consecutive_lagrange_weights,
        ) as weight_function:
            result = threshold_snip.verify_full(
                self.cfg,
                bundle,
                threshold_snip.deterministic_beacon("weight-reuse"),
            )
        self.assertTrue(result.result.accepted)
        self.assertEqual(weight_function.call_count, 1)
        self.assertGreater(result.public_weight_seconds, 0.0)

    def test_different_verification_sets_share_only_public_interpolation(self) -> None:
        protocols = (
            VMaskProtocol(
                self.cfg,
                dimension=2,
                max_clients=4,
                committee_ids=(11, 22, 33, 44, 55),
                verification_ids=(44, 11, 33),
            ),
            VMaskProtocol(
                self.cfg,
                dimension=2,
                max_clients=4,
                committee_ids=(11, 22, 33, 44, 55),
                verification_ids=(22, 55, 11),
            ),
        )
        for index, protocol in enumerate(protocols):
            result = protocol.execute_round(
                sid=f"selected-set-{index}",
                q_updates=((1, -2),),
                seeds=(91500 + index,),
            )
            self.assertTrue(result.released)
            self.assertEqual(result.accepted_clients, [0])
            self.assertTrue(result.aggregate_equality)
        info = threshold_snip.public_interpolation_cache_info()
        self.assertEqual(len(info), 1)


if __name__ == "__main__":
    unittest.main()
