from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import secrets
import sys
import time
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "src"))
from vmask_exp.protocol import prio_snip_reference as ref
from vmask_exp.protocol import threshold_snip as snip
from vmask_exp.protocol import pipeline as native
from vmask_exp.protocol.config import profile_p80, profile_p100, protocol_parameter_digest

COMMITTEE = (11, 22, 33, 44, 55)
SELECTED = (44, 11, 33)
MODES = ("range_only", "arithmetic_detached", "full_vmask")
ORIGINAL_BUILDER = ref.build_vmask_circuit
ORIGINAL_ADAPTER = ref.adapted_input_share

@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    x: int
    z: int
    y: int
    recovery_z: int
    accepted: tuple[bool, bool, bool]

CASES = (
    Case("positive", 15, 0, 15, 0, (True, True, True)),
    Case("binding_attack", 15, 0, 15, -1, (True, True, False)),
    Case("arithmetic_diagnostic", 15, 0, 16, 0, (True, False, False)),
    Case("old_proof_alternate_recovery", 15, 0, 15, 1, (True, True, False)),
    Case("fresh_legal_alternate", 14, 1, 15, 1, (True, True, True)),
    Case("range_negative", 16, 0, 16, 0, (False, False, False)),
)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def plain(value):
    if dataclasses.is_dataclass(value): return plain(dataclasses.asdict(value))
    if isinstance(value, bytes): return value.hex()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value

def source_hashes():
    return {
        p.relative_to(REPO).as_posix(): sha(p.read_bytes())
        for p in sorted((REPO / "src").rglob("*.py"))
    }

def range_builder(cfg, statement):

    d = len(statement.y)
    layout = ref.witness_layout(cfg, d)
    circuit = ref.ArithmeticCircuit(cfg.field_modulus, layout.input_count)
    inputs = [circuit.input(i) for i in range(layout.input_count)]
    one = circuit.constant(1)
    def add_many(nodes):
        result = nodes[0]
        for node in nodes[1:]: result = circuit.add(result, node)
        return result
    for j in range(d):
        x_shifted = circuit.add(inputs[layout.x.start+j], circuit.constant(cfg.amax))
        low = [inputs[layout.x_bits.start+j*layout.b_a+b] for b in range(layout.b_a)]
        high = [inputs[layout.x_complement_bits.start+j*layout.b_a+b] for b in range(layout.b_a)]
        for bit in low + high:
            circuit.constrain_zero(circuit.mul(bit, circuit.sub(bit, one)))
        low_sum = add_many([circuit.scale(n, 1 << b) for b, n in enumerate(low)])
        high_sum = add_many([circuit.scale(n, 1 << b) for b, n in enumerate(high)])
        circuit.constrain_zero(circuit.sub(x_shifted, low_sum))
        complement = circuit.sub(circuit.constant(2*cfg.amax), x_shifted)
        circuit.constrain_zero(circuit.sub(complement, high_sum))
    assert circuit.multiplication_count == 2*layout.b_a*d
    assert circuit.constraint_count == (2*layout.b_a+2)*d
    return circuit, layout

def detached_adapter(packet, coefficient, layout, q):
    assert packet.non_designated_additive_share is not None
    raw = packet.non_designated_additive_share
    non_z_count = len(layout.non_z_indices)
    if len(raw) != non_z_count + layout.dimension:
        raise ValueError("synthetic detached input length mismatch")
    base = dataclasses.replace(packet, non_designated_additive_share=raw[:non_z_count])
    normal, _ = ORIGINAL_ADAPTER(base, coefficient, layout, q)
    result = list(normal)
    claimed_z = tuple(int(v) % q for v in raw[non_z_count:])
    for index, value in zip(layout.z_indices, claimed_z): result[index] = value
    return tuple(result), claimed_z

@contextmanager
def mode_scope(mode):
    assert ref.build_vmask_circuit is ORIGINAL_BUILDER
    assert ref.adapted_input_share is ORIGINAL_ADAPTER
    if mode == "range_only":
        with mock.patch.object(ref, "build_vmask_circuit", range_builder): yield
    elif mode == "arithmetic_detached":
        with mock.patch.object(ref, "adapted_input_share", detached_adapter): yield
    else:
        assert mode == "full_vmask"
        yield
    assert ref.build_vmask_circuit is ORIGINAL_BUILDER
    assert ref.adapted_input_share is ORIGINAL_ADAPTER

def vector(value, d): return (int(value),) + (0,)*(d-1)

def witness_for(cfg, case, d):
    limbs = (vector(case.z, d),) + ((0,)*d,)*(cfg.limbs-1)
    return ref.VMaskWitnessValues(x=vector(case.x, d), limbs=limbs,
        shifted=tuple(tuple(v+cfg.r0 for v in row) for row in limbs),
        delta=(0,)*d, z=vector(case.z, d))

def interpolate_at(xs, ys, target, q):
    total = 0
    for i, x in enumerate(xs):
        coefficient = 1
        for j, other in enumerate(xs):
            if i != j: coefficient = coefficient * (target-other) * pow(x-other, -1, q) % q
        total = (total + coefficient*ys[i]) % q
    return total

def audit_shares(cfg, bundle, expected_z):
    packets = {p.committee_id:p for p in bundle.packets}
    selected = [packets[i] for i in SELECTED]
    xs = [p.share_x for p in selected]
    recovered = ref.reconstruct_shamir_vector([p.designated_shamir_share for p in selected], xs, cfg.field_modulus)
    assert tuple(recovered) == tuple(v % cfg.field_modulus for v in expected_z)
    for j in range(len(expected_z)):
        ys = [p.designated_shamir_share[j] for p in selected]
        for p in bundle.packets:
            assert interpolate_at(xs, ys, p.share_x, cfg.field_modulus) == p.designated_shamir_share[j]
    return tuple(recovered)

def make_bundle(cfg, statement, case, mode, seed, events):
    witness = witness_for(cfg, case, len(statement.y))
    started = time.perf_counter()
    bundle = snip.prove_full(cfg, statement, witness, seed)
    selected_points = tuple(COMMITTEE.index(i)+1 for i in SELECTED)
    coefficients = dict(zip(SELECTED, ref.lagrange_coefficients_zero(selected_points, cfg.field_modulus)))
    change = vector(case.recovery_z-case.z, len(statement.y))
    packets = []
    for p in bundle.packets:
        additive = p.non_designated_additive_share
        if mode == "arithmetic_detached" and p.committee_id in SELECTED:
            independent_z = tuple(coefficients[p.committee_id]*v % cfg.field_modulus for v in p.designated_shamir_share)
            additive = tuple(additive) + independent_z
        packets.append(dataclasses.replace(p, non_designated_additive_share=additive,
            designated_shamir_share=tuple((v+delta) % cfg.field_modulus for v,delta in zip(p.designated_shamir_share,change))))
    digest = ref.statement_digest(statement)
    commitments = tuple(ref.commit_packet(digest,p,cfg.field_modulus) for p in packets)
    bundle = dataclasses.replace(bundle,packets=tuple(packets),commitments=commitments)
    secret = audit_shares(cfg,bundle,vector(case.recovery_z,len(statement.y)))
    for p,c in zip(bundle.packets,bundle.commitments):
        assert ref.commit_packet(digest,p,cfg.field_modulus) == c
        assert p.share_x == COMMITTEE.index(p.committee_id)+1
    events.append(dict(event="final_packet_commitments_frozen",order=0,monotonic_ns=time.monotonic_ns(),
        statement_hash=digest.hex(),commitments=[c.hex() for c in commitments],
        payload_hashes=[sha(ref.serialize_packet_payload(p,cfg.field_modulus)) for p in packets]))
    return bundle,time.perf_counter()-started,witness,secret

def cfg_for(profile): return (profile_p80 if profile=="P80" else profile_p100)(committee_size=5,threshold=3,kappa=14)

def run_component(profile,d,case,mode,repetition):
    cfg=cfg_for(profile);cfg.validate(d,20)
    sid=f"binding-{profile}-d{d}-{case.name}-{mode}-r{repetition}"
    statement=ref.PrioStatement(theta_digest=protocol_parameter_digest(cfg,dimension=d,max_clients=20),
        sid=sid,committee_ids=COMMITTEE,verification_ids=SELECTED,client_id=0,y=vector(case.y,d))
    events=[];seed=931000+1000*repetition+sum(map(ord,case.name))
    with mode_scope(mode):
        bundle,prove_seconds,witness,recovery_secret=make_bundle(cfg,statement,case,mode,seed,events)
        beacon=secrets.token_bytes(32)
        events.append(dict(event="fresh_beacon_generated",order=1,monotonic_ns=time.monotonic_ns(),beacon=beacon.hex()))
        metrics=snip.verify_full(cfg,bundle,beacon)
    assert events[0]['order'] < events[1]['order'] and events[0]['monotonic_ns'] <= events[1]['monotonic_ns']
    expected=case.accepted[MODES.index(mode)]
    assert metrics.result.packet_openings_valid, (sid,"opening failure confounds binding")
    actual=metrics.result.accepted
    assert actual == expected,(sid,expected,plain(metrics.result))
    centered_z=[v-cfg.field_modulus if v>=cfg.field_modulus//2 else v for v in recovery_secret]
    decoded=[native.VMaskProtocol._center_mod((y-z)%cfg.modulus,cfg.modulus) for y,z in zip(statement.y,centered_z)]
    if case.name=="binding_attack" and mode=="full_vmask":
        assert metrics.result.polynomial_identity_residual==0
        assert metrics.result.constraint_residual!=0
    return dict(kind="component",profile=profile,dimension=d,case=case.name,mode=mode,repetition=repetition,
        seed=seed,statement=plain(statement),witness=plain(witness),claimed_x=case.x,claimed_z=case.z,
        actual_recovery_z=case.recovery_z,events=events,expected_accepted=expected,actual_accepted=actual,
        openings_valid=metrics.result.packet_openings_valid,polynomial_residual=metrics.result.polynomial_identity_residual,
        relation_residual=metrics.result.constraint_residual,decoded_if_accepted=decoded if actual else None,
        decoded_test_oracle=decoded,prove_seconds=prove_seconds,check_seconds=metrics.total_seconds,
        synthetic=mode!="full_vmask",pass_=True)

def as_list(value): return native.VMaskProtocol._to_int_list(value)

def run_native(case,mode,repetition):
    cfg=cfg_for("P80");d=1;events=[];created=[];client_events={};observed=[]
    protocol=native.VMaskProtocol(cfg,d,20,committee_ids=COMMITTEE,verification_ids=SELECTED)
    sid=f"binding-native-{case.name}-{mode}-r{repetition}"
    zero=Case("honest_control",0,0,0,0,(True,True,True))
    def make_submission(**kwargs):
        current=case if kwargs['client_id']==0 else zero
        statement=protocol._formal_statement(sid=kwargs['sid'],client_id=kwargs['client_id'],y=vector(current.y,d))
        entries=[]
        bundle,elapsed,witness,secret=make_bundle(cfg,statement,current,mode,int(kwargs['seed']),entries)
        client_events[kwargs['client_id']]=entries
        created.append(bundle)
        e=native.VerificationEvidence(relation_digest=protocol.theta_digest,statement_digest=ref.statement_digest(statement),
            packet_commitments=bundle.commitments,verifier_packets=bundle.packets,
            multiplication_count=bundle.multiplication_count,constraint_count=bundle.constraint_count,
            witness_length=bundle.witness_length,backend=snip.BACKEND_NAME)
        return native.ClientSubmission(gamma=native.ProtectedSubmission(sid=statement.sid,client_id=statement.client_id,y=list(statement.y)),
            evidence=e,witness_designated_output=list(witness.z),relation_satisfied=bundle.relation_satisfied,snip_prove_seconds=elapsed)
    def fresh_beacon():
        client_id=len(created)-1;bundle=created[-1]
        digest=ref.statement_digest(bundle.statement)
        assert all(ref.commit_packet(digest,p,cfg.field_modulus)==c for p,c in zip(bundle.packets,bundle.commitments))
        b=secrets.token_bytes(32)
        client_events[client_id].append(dict(event="fresh_beacon_generated",order=1,monotonic_ns=time.monotonic_ns(),beacon=b.hex()))
        return b
    original_reconstruct=native.reconstruct_shamir_vector
    def reconstruct(shares,xs,q):

        accept_target=case.accepted[MODES.index(mode)]
        identities=range(20) if accept_target else range(1,20)
        expected=[]
        for verifier_id in SELECTED:
            expected.append(tuple(sum(next(p for p in created[i].packets if p.committee_id==verifier_id).designated_shamir_share[j] for i in identities)%q for j in range(d)))
        assert tuple(map(tuple,shares))==tuple(expected)
        observed.append(dict(expected_original_share_sum=plain(expected),actual_reconstruction_shares=plain(shares),share_points=plain(xs)))
        return original_reconstruct(shares,xs,q)
    updates=(vector(case.x,d),)+((0,)*d,)*19
    with mode_scope(mode),mock.patch.object(protocol,"make_submission",side_effect=make_submission),mock.patch.object(native,"generate_challenge_seed",side_effect=fresh_beacon),mock.patch.object(native,"reconstruct_shamir_vector",side_effect=reconstruct):
        result=protocol.execute_round(sid=sid,q_updates=updates,seeds=tuple(941000+1000*repetition+i for i in range(20)))
    accept_target=case.accepted[MODES.index(mode)]
    expected_ids=list(range(20)) if accept_target else list(range(1,20))
    expected_value=case.y-case.recovery_z if accept_target else 0
    assert result.released and result.accepted_clients==expected_ids
    assert result.rejected_clients==([] if accept_target else [0])
    assert as_list(result.recovered_q_sum)==[expected_value]
    assert len(observed)==1 and len(created)==20
    for entries in client_events.values():
        assert len(entries)==2 and entries[0]['order'] < entries[1]['order'] and entries[0]['monotonic_ns'] <= entries[1]['monotonic_ns']
    expected_equality=not accept_target or expected_value==case.x
    assert result.aggregate_equality==expected_equality
    return dict(kind="native_round",profile="P80",dimension=1,mode=mode,case=case.name,repetition=repetition,
        params=dict(N_t=20,N_max=20,n=5,tau=3,kappa=14),accepted_clients=result.accepted_clients,rejected_clients=result.rejected_clients,
        released=result.released,aggregate_output=as_list(result.recovered_q_sum),aggregate_equality=result.aggregate_equality,
        expected_output=[expected_value],recovery_observer=observed,client_events=client_events,
        timing=plain(result.timing),synthetic=mode!="full_vmask",pass_=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true",
        help="run the P80 binding counterexample in all three modes")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    results_dir=args.output_dir.resolve();results_dir.mkdir(parents=True,exist_ok=True)
    if (results_dir/"records.jsonl").exists():raise FileExistsError("Refusing to overwrite raw records; preserve existing run")
    before=source_hashes();records=[];started=time.time()
    environment=dict(python=sys.version,platform=platform.platform(),flint_version=__import__('flint').__version__,source_hashes_before=before)
    (results_dir/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    schedule=[]
    if args.smoke: schedule=[("P80",1,CASES[1],m,0) for m in MODES]
    else:
        for profile in ["P80","P100"]:
            schedule.extend((profile,1,c,m,r) for c in CASES for m in MODES for r in range(3))
        schedule.extend(("P80",10,c,m,r) for c in [CASES[0],CASES[1],CASES[4]] for m in MODES for r in range(3))
    with (results_dir/'records.jsonl').open('x',encoding='utf-8') as stream:
        for profile,d,case,mode,repetition in schedule:
            record=run_component(profile,d,case,mode,repetition);records.append(record)
            stream.write(json.dumps(record)+"\n");stream.flush()
            print('PASS component',profile,d,case.name,mode,repetition,flush=True)
        if not args.smoke:
            for case in [CASES[0],CASES[1],CASES[4]]:
                for mode in MODES:
                    for repetition in range(3):
                        record=run_native(case,mode,repetition);records.append(record)
                        stream.write(json.dumps(record)+"\n");stream.flush()
                        print('PASS native',case.name,mode,repetition,flush=True)
    assert before==source_hashes()
    with (results_dir/'component_results.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['profile','dimension','case','mode','repetition','expected_accepted','actual_accepted','openings_valid','polynomial_residual','relation_residual','decoded_if_accepted','pass_']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            if r['kind']=='component':writer.writerow({k:r[k] for k in fields})
    with (results_dir/'native_results.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['case','mode','repetition','released','accepted_count','aggregate_output','aggregate_equality','pass_']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            if r['kind']=='native_round':writer.writerow({**{k:r[k] for k in fields if k!='accepted_count'},'accepted_count':len(r['accepted_clients'])})
    summary=dict(status='complete',component_cases=sum(r['kind']=='component' for r in records),native_rounds=sum(r['kind']=='native_round' for r in records),
        all_passed=all(r['pass_'] for r in records),source_files_unchanged=True,elapsed_seconds=time.time()-started,
        script_sha256=sha(Path(__file__).read_bytes()),mode_functions_restored=ref.build_vmask_circuit is ORIGINAL_BUILDER and ref.adapted_input_share is ORIGINAL_ADAPTER)
    (results_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
