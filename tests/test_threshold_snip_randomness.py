from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

from vmask_exp.protocol import pipeline, threshold_snip
from vmask_exp.protocol.commitment import generate_challenge_seed
from vmask_exp.protocol.config import profile_p80
from vmask_exp.protocol.prio_snip_reference import serialize_packet_payload


class SecureRandomnessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = profile_p80(committee_size=5, threshold=3, kappa=2)
        self.protocol = pipeline.VMaskProtocol(
            self.cfg, dimension=2, max_clients=4,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        self.updates = ((1, -2), (-3, 4))

    def _assert_real_accept(self, submission):
        metrics = threshold_snip.verify_full(
            self.cfg, self.protocol._formal_bundle(submission),
            generate_challenge_seed(),
        )
        self.assertTrue(metrics.result.accepted)
        self.assertTrue(metrics.result.packet_openings_valid)
        self.assertEqual(metrics.result.polynomial_identity_residual, 0)
        self.assertEqual(metrics.result.constraint_residual, 0)
        self.assertEqual(
            tuple(t.committee_id for t in metrics.verifier_timings),
            (44, 11, 33),
        )
        return metrics

    def test_secure_default_uses_system_random_for_mask_and_proof(self):


        original_system_random = pipeline.secrets.SystemRandom
        with patch.object(
            pipeline.secrets, "SystemRandom", wraps=original_system_random,
        ) as secure_source, patch.object(
            pipeline.random, "Random",
            side_effect=AssertionError("seeded MT is forbidden in secure default"),
        ):
            submission = self.protocol.make_submission(
                sid="secure-default-real-check", client_id=0,
                q_update=self.updates[0],
            )
        self.assertEqual(secure_source.call_count, 2)
        self._assert_real_accept(submission)

    def test_same_explicit_seed_still_uses_fresh_os_salts(self):
        original_token_bytes = threshold_snip.secrets.token_bytes
        observed = []

        def observe_os_bytes(length):
            value = original_token_bytes(length)
            observed.append((length, value))
            return value

        arguments = dict(
            sid="seeded-but-fresh-salts", client_id=0,
            q_update=self.updates[0], seed=91501,
        )

        with patch.object(
            threshold_snip.secrets, "token_bytes", side_effect=observe_os_bytes,
        ) as salt_source:
            first = self.protocol.make_submission(**arguments)
            second = self.protocol.make_submission(**arguments)
        self.assertEqual(salt_source.call_count, 2 * self.cfg.committee_size)
        self.assertTrue(all(length == len(value) == 32 for length, value in observed))
        packets = first.evidence.verifier_packets + second.evidence.verifier_packets
        self.assertEqual(
            [p.commitment_randomness for p in packets],
            [value for _, value in observed],
        )
        self.assertEqual(first.gamma, second.gamma)
        for old, new in zip(first.evidence.verifier_packets, second.evidence.verifier_packets):
            self.assertEqual(
                serialize_packet_payload(old, self.cfg.field_modulus),
                serialize_packet_payload(new, self.cfg.field_modulus),
            )


        self.assertNotEqual(
            first.evidence.packet_commitments, second.evidence.packet_commitments,
        )
        self._assert_real_accept(first)
        self._assert_real_accept(second)

    def test_salt_payload_and_statement_tampering_are_rejected(self):
        submission = self.protocol.make_submission(
            sid="salt-binding-real-check", client_id=0,
            q_update=self.updates[0],
        )
        self._assert_real_accept(submission)
        bundle = self.protocol._formal_bundle(submission)
        index = next(i for i, p in enumerate(bundle.packets) if p.committee_id == 44)
        selected = bundle.packets[index]
        changed_salt = bytes([selected.commitment_randomness[0] ^ 1]) + selected.commitment_randomness[1:]
        changed_share = (
            (selected.designated_shamir_share[0] + 1) % self.cfg.field_modulus,
            *selected.designated_shamir_share[1:],
        )
        variants = {}
        for name, changed in (
            ("salt", dataclasses.replace(selected, commitment_randomness=changed_salt)),
            ("payload", dataclasses.replace(selected, designated_shamir_share=changed_share)),
        ):
            packets = list(bundle.packets)
            packets[index] = changed
            variants[name] = dataclasses.replace(bundle, packets=tuple(packets))
        variants["statement"] = dataclasses.replace(
            bundle, statement=dataclasses.replace(bundle.statement, sid="another-session"),
        )
        for name, candidate in variants.items():
            with self.subTest(changed=name):
                metrics = threshold_snip.verify_full(
                    self.cfg, candidate, generate_challenge_seed(),
                )
                self.assertFalse(metrics.result.accepted)
                self.assertFalse(metrics.result.packet_openings_valid)
                self.assertEqual(metrics.verifier_timings, ())

    def test_salt_entropy_failure_has_no_fallback_check_or_release(self):


        with patch.object(
            threshold_snip.secrets, "token_bytes", side_effect=OSError("salt entropy unavailable"),
        ), patch.object(
            threshold_snip.reference, "random_bytes",
            side_effect=AssertionError("no seeded salt fallback"),
        ), patch.object(
            pipeline, "verify_full", wraps=pipeline.verify_full,
        ) as checking, patch.object(
            pipeline, "reconstruct_shamir_vector", wraps=pipeline.reconstruct_shamir_vector,
        ) as recovery:
            with self.assertRaisesRegex(OSError, "salt entropy unavailable"):
                self.protocol.execute_round(
                    sid="salt-source-failure", q_updates=self.updates,
                    seeds=(91502, 91503),
                )
        self.assertEqual(checking.call_count, 0)
        self.assertEqual(recovery.call_count, 0)

    def test_mask_entropy_failure_has_no_seeded_fallback_or_release(self):
        with patch.object(
            pipeline.secrets, "SystemRandom", side_effect=OSError("mask entropy unavailable"),
        ), patch.object(
            pipeline.random, "Random", side_effect=AssertionError("no seeded mask fallback"),
        ), patch.object(
            pipeline, "verify_full", wraps=pipeline.verify_full,
        ) as checking, patch.object(
            pipeline, "reconstruct_shamir_vector", wraps=pipeline.reconstruct_shamir_vector,
        ) as recovery:
            with self.assertRaisesRegex(OSError, "mask entropy unavailable"):
                self.protocol.execute_round(
                    sid="mask-source-failure", q_updates=self.updates,
                )
        self.assertEqual(checking.call_count, 0)
        self.assertEqual(recovery.call_count, 0)

    def test_proof_entropy_failure_after_real_mask_has_no_fallback_or_release(self):
        original_system_random = pipeline.secrets.SystemRandom
        calls = []

        def secure_mask_then_proof_failure():
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                return original_system_random()
            raise OSError("proof entropy unavailable")

        with patch.object(
            pipeline.secrets, "SystemRandom", side_effect=secure_mask_then_proof_failure,
        ), patch.object(
            pipeline.random, "Random", side_effect=AssertionError("no seeded proof fallback"),
        ), patch.object(
            pipeline, "verify_full", wraps=pipeline.verify_full,
        ) as checking, patch.object(
            pipeline, "reconstruct_shamir_vector", wraps=pipeline.reconstruct_shamir_vector,
        ) as recovery:
            with self.assertRaisesRegex(OSError, "proof entropy unavailable"):
                self.protocol.execute_round(
                    sid="proof-source-failure", q_updates=self.updates,
                )
        self.assertEqual(calls, [1, 2])
        self.assertEqual(checking.call_count, 0)
        self.assertEqual(recovery.call_count, 0)


if __name__ == "__main__":
    unittest.main()
