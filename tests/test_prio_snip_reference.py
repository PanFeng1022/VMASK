from __future__ import annotations

import dataclasses
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmask_exp.protocol.prio_snip_reference import (
    PrioProofShare,
    PrioStatement,
    VMaskWitnessValues,
    cost_estimate,
    prove,
    replace_packet,
    verify,
)
from vmask_exp.protocol.config import ProtocolConfig, protocol_parameter_digest


class PrioSnipReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ProtocolConfig(
            modulus=2**32,
            amax=1,
            r0=1,
            limbs=2,
            beta=3,
            committee_size=5,
            threshold=3,
            kappa=1,
        )
        self.dimension = 2
        self.committee_ids = (11, 22, 33, 44, 55)
        self.verification_ids = (44, 11, 33)
        self.witness = VMaskWitnessValues(
            x=(1, -1),
            limbs=((1, -1), (-1, 1)),
            shifted=((2, 0), (0, 2)),
            delta=(1, 0),
            z=(-2, 2),
        )
        self.statement = PrioStatement(
            theta_digest=protocol_parameter_digest(
                self.cfg,
                dimension=self.dimension,
                max_clients=20,
            ),
            sid="prio-adapter-test",
            committee_ids=self.committee_ids,
            verification_ids=self.verification_ids,
            client_id=7,
            y=(self.cfg.modulus - 1, 1),
        )
        self.bundle = prove(self.cfg, self.statement, self.witness, seed=90210)

    def beacon(self, index: int = 0) -> bytes:
        return hashlib.sha256(f"beacon-{index}".encode("ascii")).digest()

    def packet_index(self, committee_id: int) -> int:
        return self.committee_ids.index(committee_id)

    def test_honest_proof_accepts_and_binds_designated_output(self) -> None:
        result = verify(self.cfg, self.bundle, self.beacon())
        expected = tuple(value % self.cfg.field_modulus for value in self.witness.z)
        self.assertTrue(self.bundle.relation_satisfied)
        self.assertTrue(result.accepted)
        self.assertTrue(result.packet_openings_valid)
        self.assertEqual(result.polynomial_identity_residual, 0)
        self.assertEqual(result.constraint_residual, 0)
        self.assertEqual(result.reconstructed_designated_output, expected)
        self.assertEqual(result.adapter_output, expected)

    def test_invalid_c1_c6_witness_rejects(self) -> None:
        invalid = dataclasses.replace(
            self.witness,
            z=(self.witness.z[0] + 1, self.witness.z[1]),
        )
        bundle = prove(self.cfg, self.statement, invalid, seed=90211)
        self.assertFalse(bundle.relation_satisfied)
        for index in range(16):
            self.assertFalse(verify(self.cfg, bundle, self.beacon(index)).accepted)

    def test_selected_designated_share_replacement_before_challenge_rejects(self) -> None:
        index = self.packet_index(self.verification_ids[0])
        packet = self.bundle.packets[index]
        changed = list(packet.designated_shamir_share)
        changed[0] = (changed[0] + 1) % self.cfg.field_modulus
        packet = dataclasses.replace(packet, designated_shamir_share=tuple(changed))
        forged = replace_packet(self.cfg, self.bundle, index, packet, recommit=True)
        for beacon_index in range(16):
            result = verify(self.cfg, forged, self.beacon(beacon_index))
            self.assertFalse(result.accepted)
            self.assertEqual(result.reconstructed_designated_output, result.adapter_output)

    def test_proof_polynomial_replacement_before_challenge_rejects(self) -> None:
        index = self.packet_index(self.verification_ids[1])
        packet = self.bundle.packets[index]
        proof = packet.proof_share
        h = list(proof.h_coefficients)
        h[0] = (h[0] + 1) % self.cfg.field_modulus
        packet = dataclasses.replace(
            packet,
            proof_share=dataclasses.replace(proof, h_coefficients=tuple(h)),
        )
        forged = replace_packet(self.cfg, self.bundle, index, packet, recommit=True)
        for beacon_index in range(16):
            self.assertFalse(verify(self.cfg, forged, self.beacon(beacon_index)).accepted)

    def test_invalid_beaver_triple_before_challenge_rejects(self) -> None:
        index = self.packet_index(self.verification_ids[2])
        packet = self.bundle.packets[index]
        proof = packet.proof_share
        packet = dataclasses.replace(
            packet,
            proof_share=dataclasses.replace(
                proof,
                triple_c=(proof.triple_c + 1) % self.cfg.field_modulus,
            ),
        )
        forged = replace_packet(self.cfg, self.bundle, index, packet, recommit=True)
        for beacon_index in range(16):
            self.assertFalse(verify(self.cfg, forged, self.beacon(beacon_index)).accepted)

    def test_zero_proof_material_attack_rejects(self) -> None:
        forged = self.bundle
        h_length = 2 * self.bundle.multiplication_count + 1
        zero_proof = PrioProofShare(0, 0, (0,) * h_length, 0, 0, 0)
        for committee_id in self.verification_ids:
            index = self.packet_index(committee_id)
            packet = dataclasses.replace(forged.packets[index], proof_share=zero_proof)
            forged = replace_packet(self.cfg, forged, index, packet, recommit=True)
        for beacon_index in range(16):
            self.assertFalse(verify(self.cfg, forged, self.beacon(beacon_index)).accepted)

    def test_postcommit_packet_change_rejects_at_opening(self) -> None:
        index = self.packet_index(self.verification_ids[0])
        packet = self.bundle.packets[index]
        changed = list(packet.designated_shamir_share)
        changed[-1] = (changed[-1] + 1) % self.cfg.field_modulus
        packet = dataclasses.replace(packet, designated_shamir_share=tuple(changed))
        forged = replace_packet(self.cfg, self.bundle, index, packet, recommit=False)
        result = verify(self.cfg, forged, self.beacon())
        self.assertFalse(result.accepted)
        self.assertFalse(result.packet_openings_valid)

    def test_cost_formulas_match_constructed_circuit(self) -> None:
        costs = cost_estimate(self.cfg, self.dimension)
        self.assertEqual(costs.witness_field_elements, self.bundle.witness_length)
        self.assertEqual(costs.multiplication_gates, self.bundle.multiplication_count)
        self.assertEqual(costs.constraints, self.bundle.constraint_count)
        self.assertEqual(costs.proof_share_field_elements, 2 * self.bundle.multiplication_count + 6)


if __name__ == "__main__":
    unittest.main()
