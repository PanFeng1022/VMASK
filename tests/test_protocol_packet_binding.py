from __future__ import annotations

import dataclasses
import unittest

from vmask_exp.protocol.config import profile_p80
from vmask_exp.protocol.pipeline import VMaskProtocol
from vmask_exp.protocol.prio_snip_reference import (
    PrioPacket,
    PrioProofShare,
    commit_packet,
    reconstruct_shamir_vector,
    statement_digest,
)
from vmask_exp.protocol.threshold_snip import BACKEND_NAME
from vmask_exp.protocol.serialization import (
    component_byte_counts,
    measure_submission_components,
)


def as_list(values) -> list[int]:
    if hasattr(values, "detach"):
        return [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
    return [int(value) for value in values]


class CompleteThresholdSnipIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dimension = 10
        self.cfg = profile_p80(committee_size=5, threshold=3, kappa=1)
        self.protocol = VMaskProtocol(
            self.cfg,
            dimension=self.dimension,
            max_clients=4,
            committee_ids=(11, 22, 33, 44, 55),
            verification_ids=(44, 11, 33),
        )
        self.updates = (
            tuple((index % 7) - 3 for index in range(self.dimension)),
            tuple(2 - (index % 5) for index in range(self.dimension)),
        )
        self.seeds = (91001, 91002)

    def _execute_with_transform(self, transform):
        original = self.protocol.make_submission
        captured = []

        def patched(**kwargs):
            submission = original(**kwargs)
            if kwargs["client_id"] == 0:
                submission = transform(submission)
            captured.append(submission)
            return submission

        self.protocol.make_submission = patched
        try:
            result = self.protocol.execute_round(
                sid="native-full-snip-tamper",
                q_updates=self.updates,
                seeds=self.seeds,
            )
        finally:
            self.protocol.make_submission = original
        return result, captured

    def _replace_packets(self, submission, packets, *, recommit: bool):
        statement = self.protocol._formal_statement(
            sid=submission.gamma.sid,
            client_id=submission.gamma.client_id,
            y=submission.gamma.y,
        )
        if recommit:
            digest = statement_digest(statement)
            commitments = tuple(
                commit_packet(digest, packet, self.cfg.field_modulus)
                for packet in packets
            )
        else:
            commitments = submission.evidence.packet_commitments
        evidence = dataclasses.replace(
            submission.evidence,
            verifier_packets=tuple(packets),
            packet_commitments=commitments,
        )
        return dataclasses.replace(submission, evidence=evidence)

    def test_t1_honest_client_executes_complete_protocol(self) -> None:
        result = self.protocol.execute_round(
            sid="native-full-snip-t1",
            q_updates=(self.updates[0],),
            seeds=(self.seeds[0],),
        )
        self.assertTrue(result.released)
        self.assertEqual(result.accepted_clients, [0])
        self.assertEqual(result.rejected_clients, [])
        self.assertTrue(result.aggregate_equality)
        self.assertEqual(as_list(result.recovered_q_sum), list(self.updates[0]))
        self.assertGreater(result.timing.client_prove_seconds[0], 0.0)
        self.assertGreater(result.timing.client_check_seconds[0], 0.0)
        self.assertEqual(self.protocol.formal_snip_backend, BACKEND_NAME)

    def test_t2_relation_violating_amm_is_rejected(self) -> None:
        result = self.protocol.execute_round(
            sid="native-full-snip-t2",
            q_updates=self.updates,
            seeds=self.seeds,
            malformed_clients=(0,),
            malformed_direction=(1,) * self.dimension,
        )
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual(result.rejected_clients, [0])
        self.assertEqual(result.relation_failures, 1)
        self.assertTrue(result.aggregate_equality)
        self.assertEqual(as_list(result.recovered_q_sum), list(self.updates[1]))

    def test_t3_recommitted_zero_proof_and_mask_shares_are_rejected(self) -> None:
        def transform(submission):
            multiplication_count = submission.evidence.multiplication_count
            zero_proof = PrioProofShare(
                f_zero=0,
                g_zero=0,
                h_coefficients=(0,) * (2 * multiplication_count + 1),
                triple_a=0,
                triple_b=0,
                triple_c=0,
            )
            packets = []
            for packet in submission.evidence.verifier_packets:
                proof = (
                    zero_proof
                    if packet.committee_id in self.protocol.verification_ids
                    else None
                )
                packets.append(
                    dataclasses.replace(
                        packet,
                        designated_shamir_share=(0,) * self.dimension,
                        proof_share=proof,
                    )
                )
            return self._replace_packets(submission, packets, recommit=True)

        result, _ = self._execute_with_transform(transform)
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual(result.rejected_clients, [0])
        self.assertEqual(result.relation_failures, 1)
        self.assertTrue(result.aggregate_equality)

    def test_t4_recommitted_designated_share_change_is_rejected(self) -> None:
        def transform(submission):
            packets = [
                dataclasses.replace(
                    packet,
                    designated_shamir_share=tuple(
                        (value + 1) % self.cfg.field_modulus
                        for value in packet.designated_shamir_share
                    ),
                )
                for packet in submission.evidence.verifier_packets
            ]
            return self._replace_packets(submission, packets, recommit=True)

        result, _ = self._execute_with_transform(transform)
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual(result.rejected_clients, [0])
        self.assertEqual(result.relation_failures, 1)
        self.assertTrue(result.aggregate_equality)

    def test_t5_packet_change_without_opening_update_is_rejected(self) -> None:
        def transform(submission):
            packets = list(submission.evidence.verifier_packets)
            selected_id = self.protocol.verification_ids[0]
            packet_index = self.protocol.committee_ids.index(selected_id)
            packet = packets[packet_index]
            proof = packet.proof_share
            changed_h = list(proof.h_coefficients)
            changed_h[0] = (changed_h[0] + 1) % self.cfg.field_modulus
            packets[packet_index] = dataclasses.replace(
                packet,
                proof_share=dataclasses.replace(
                    proof,
                    h_coefficients=tuple(changed_h),
                ),
            )
            return self._replace_packets(submission, packets, recommit=False)

        result, _ = self._execute_with_transform(transform)
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual(result.rejected_clients, [0])
        self.assertEqual(result.packet_failures, 1)
        self.assertTrue(result.aggregate_equality)

    def test_t6_two_honest_clients_aggregate_correctly(self) -> None:
        result = self.protocol.execute_round(
            sid="native-full-snip-t6",
            q_updates=self.updates,
            seeds=self.seeds,
        )
        expected = [
            first + second
            for first, second in zip(self.updates[0], self.updates[1])
        ]
        self.assertTrue(result.released)
        self.assertEqual(result.accepted_clients, [0, 1])
        self.assertTrue(result.aggregate_equality)
        self.assertEqual(as_list(result.recovered_q_sum), expected)

    def test_t7_fixed_verifier_unavailable_aborts_round(self) -> None:
        available = set(self.protocol.committee_ids)
        available.remove(self.protocol.verification_ids[0])
        result = self.protocol.execute_round(
            sid="native-full-snip-t7",
            q_updates=self.updates,
            seeds=self.seeds,
            available_committee_ids=available,
        )
        self.assertFalse(result.released)
        self.assertEqual(
            result.release_reason,
            "verification_committee_unavailable",
        )
        self.assertEqual(result.timing.client_prove_seconds, ())

    def test_t8_recovery_reuses_exact_verified_packet_shares(self) -> None:
        original = self.protocol.make_submission
        captured = []

        def capture(**kwargs):
            submission = original(**kwargs)
            captured.append(submission)
            return submission

        self.protocol.make_submission = capture
        try:
            result = self.protocol.execute_round(
                sid="native-full-snip-t8",
                q_updates=self.updates,
                seeds=self.seeds,
            )
        finally:
            self.protocol.make_submission = original

        packet_maps = [
            {
                packet.committee_id: packet
                for packet in submission.evidence.verifier_packets
            }
            for submission in captured
        ]
        aggregate_shares = tuple(
            tuple(
                sum(
                    packet_map[committee_id].designated_shamir_share[coordinate]
                    for packet_map in packet_maps
                )
                % self.cfg.field_modulus
                for coordinate in range(self.dimension)
            )
            for committee_id in self.protocol.verification_ids
        )
        share_points = tuple(
            self.protocol._committee_positions[committee_id] + 1
            for committee_id in self.protocol.verification_ids
        )
        reconstructed = reconstruct_shamir_vector(
            aggregate_shares,
            share_points,
            self.cfg.field_modulus,
        )
        expected_mask = [
            self.protocol._center_mod(value, self.cfg.field_modulus)
            for value in reconstructed
        ]
        self.assertTrue(result.released)
        self.assertEqual(result.aggregate_mask, expected_mask)
        self.assertTrue(result.aggregate_equality)

    def test_backend_selection_enforces_native_verification(self) -> None:
        def transform(submission):
            evidence = dataclasses.replace(
                submission.evidence,
                backend="unsupported_backend",
            )
            return dataclasses.replace(submission, evidence=evidence)

        result, _ = self._execute_with_transform(transform)
        self.assertEqual(result.accepted_clients, [1])
        self.assertEqual(result.rejected_clients, [0])
        self.assertEqual(result.packet_failures, 1)
        self.assertTrue(result.aggregate_equality)

    def test_complete_packet_serialization_matches_analytical_counts(self) -> None:
        submission = self.protocol.make_submission(
            sid="native-full-snip-serialization",
            client_id=0,
            q_update=self.updates[0],
            seed=self.seeds[0],
        )
        measured = measure_submission_components(submission, self.cfg)
        analytical = component_byte_counts(
            self.cfg,
            dimension=self.dimension,
        )
        self.assertEqual(submission.evidence.backend, BACKEND_NAME)
        self.assertEqual(
            measured.designated_share_bytes,
            analytical.designated_share_bytes,
        )
        self.assertEqual(
            measured.backend_verification_material_bytes,
            analytical.backend_verification_material_bytes,
        )
        self.assertEqual(
            measured.packet_framing_bytes,
            analytical.packet_framing_bytes,
        )
        self.assertEqual(
            measured.residual_reconstruction_bytes,
            2 * (self.cfg.threshold - 1) * 24,
        )
        self.assertEqual(
            measured.d_e_opening_bytes,
            4 * (self.cfg.threshold - 1) * 24,
        )


if __name__ == "__main__":
    unittest.main()
