from pathlib import Path
import argparse,ast,array,ctypes,gc,hashlib,inspect,json,os,platform,secrets,sys,time,traceback
import psutil
REPO=Path(__file__).resolve().parent.parent
SHA=lambda b:hashlib.sha256(b).hexdigest()
MiB=1024**2
KEEP=[]

def job_limit(limit_mib,cpu_seconds=900):
    if limit_mib <= 0:
        return {'hard_private_commit_limit_mib':None,'cpu_limit_seconds':None,
                'enforced':False}
    if os.name != 'nt':
        return {'hard_private_commit_limit_mib':limit_mib,
                'cpu_limit_seconds':cpu_seconds,'enforced':False,
                'reason':'Windows Job Object limits are unavailable on this platform'}
    from ctypes import wintypes as w
    class BASIC(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
          ('flags',w.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
          ('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(n,ctypes.c_ulonglong) for n in ['ro','wo','oo','rb','wb','ob']]
    class EXT(ctypes.Structure):
        _fields_=[('basic',BASIC),('io',IO),('process_memory',ctypes.c_size_t),
                  ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
    k.GetCurrentProcess.restype=w.HANDLE
    h=k.CreateJobObjectW(None,None)
    info=EXT();info.basic.flags=0x100|0x2|0x2000
    info.basic.process_time=int(cpu_seconds*10_000_000);info.process_memory=int(limit_mib*MiB)
    if not h or not k.SetInformationJobObject(h,9,ctypes.byref(info),ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.AssignProcessToJobObject(h,k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    KEEP.append((k,h))
    return {'hard_private_commit_limit_mib':limit_mib,'cpu_limit_seconds':cpu_seconds,
            'windows_job':True,'enforced':True}

def peak():
    m=psutil.Process().memory_info()
    return {'rss_mib':m.rss/MiB,
            'os_peak_rss_mib':getattr(m,'peak_wset',m.rss)/MiB,
            'private_mib':getattr(m,'private',m.rss)/MiB,
            'platform_specific_peak_fields':hasattr(m,'peak_wset') and hasattr(m,'private')}
def event(name,**kw):
    print(json.dumps({'event':name,'elapsed_monotonic':time.perf_counter(),**kw}),flush=True)
def source_nodes(path):
    src=path.read_text(encoding='utf-8')
    return src,ast.parse(src)

def run(args):
    limits=job_limit(args.limit_mib)
    sys.path.insert(0,str(REPO/'src'))
    from vmask_exp.protocol.config import profile_p80,profile_p100
    from vmask_exp.protocol.pipeline import VMaskProtocol
    from vmask_exp.protocol import prio_snip_reference as ref
    cfg=(profile_p80 if args.profile=='P80' else profile_p100)(kappa=14)
    proto=VMaskProtocol(cfg,args.d,20,committee_ids=(11,22,33,44,55),verification_ids=(44,11,33))
    q=cfg.field_modulus;M=cfg.modulus;d=args.d
    rng=secrets.SystemRandom()
    psrc,ptree=source_nodes(REPO/'src/vmask_exp/protocol/pipeline.py')
    rsrc,rtree=source_nodes(REPO/'src/vmask_exp/protocol/prio_snip_reference.py')
    cls=next(n for n in ptree.body if isinstance(n,ast.ClassDef) and n.name=='VMaskProtocol')
    make=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='make_submission')
    execute=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute_round')
    adapter=next(n for n in rtree.body if isinstance(n,ast.FunctionDef) and n.name=='adapted_input_share')
    spans={}
    def compile_node(node,name,src,filename):
        spans[name]={'source':str(filename),'first_line':node.lineno,'last_line':node.end_lineno,
                     'source_sha256':SHA(ast.get_source_segment(src,node).encode()),
                     'ast':ast.dump(node,include_attributes=False)}
        return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(filename),'exec')
    assignments={}
    for n in ast.walk(make):
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in ('limbs','y_source','delta','y'):
                    if target.id=='y_source' and not isinstance(n.value,ast.ListComp):continue
                    assignments[target.id]=compile_node(n,target.id,psrc,'pipeline.py')
    for n in ast.walk(adapter):
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='designated_additive' for x in n.targets):
            convert_code=compile_node(n,'designated_conversion',rsrc,'prio_snip_reference.py')
    for n in ast.walk(execute):
        if not isinstance(n,ast.Assign):continue
        targets=n.targets
        if any(isinstance(x,ast.Name) and x.id in ('y_aggregate','z_aggregate','aggregate_mask','recovered','centered') for x in targets):
            name=targets[0].id
            if name=='y_aggregate' and not any(isinstance(x,ast.BinOp) and isinstance(x.op,ast.Mod) for x in ast.walk(n.value)):continue
            assignments[name]=compile_node(n,name,psrc,'pipeline.py')
        if any(isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='aggregate_designated_shares' for x in targets):
            assignments['shareagg']=compile_node(n,'shareagg',psrc,'pipeline.py')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    slices_path=args.output_dir/'native_slices.json'
    if slices_path.exists():
        if json.loads(slices_path.read_text(encoding='utf-8')) != spans:
            raise RuntimeError('native_slices.json does not match the current production source')
    else:
        slices_path.write_text(json.dumps(spans,indent=2),encoding='utf-8')
    env={'self':proto,'rng':rng,'q':q,'x':None,'reconstruct_shamir_vector':ref.reconstruct_shamir_vector}
    def stmt(name):exec(assignments[name],env)
    def prepare(cid):
        env['x']=[(j+7*cid)%31-15 for j in range(d)]
        a=time.perf_counter();stmt('limbs');mask=proto._compose_mask(env['limbs']);b=time.perf_counter()
        shares=ref.shamir_share_vector(mask,5,3,q,rng);c=time.perf_counter()
        env['mask']=mask;stmt('y_source');stmt('delta');stmt('y');e=time.perf_counter()
        return env['x'],mask,shares,env['y'],[b-a,c-b,e-c]
    nonce=secrets.token_hex(16)
    result={'status':'RUNNING','role':args.role,'profile':args.profile,'d':d,'repeat':args.repeat,
      'run_nonce_not_rng_seed':nonce,'random_source':'secrets.SystemRandom / OS CSPRNG; no deterministic seed',
      'contributor_set':list(range(20)),'contributor_status':'predefined valid contributors; SNIP not executed',
      'committee_ids':list(proto.committee_ids),'fixed_T_ids':list(proto.verification_ids),'share_points':[4,1,3],
      'kappa':14,'kappa_tested':False,'config':vars(cfg),'limits':limits,
      'python':sys.version,'psutil':psutil.__version__,'memory_at_import':peak(),
      'actual_processed_input_coordinates':20*d,'snip_prove_calls':0,'snip_check_calls':0}
    event('start',role=args.role,d=d,profile=args.profile,**peak())
    times=[];source_checks=[]
    if args.role=='client':
        for cid in range(20):
            x,z,shares,y,ts=prepare(cid);times.append(ts)

            assert all((y[j]-z[j])%M==x[j]%M for j in range(d))
            del x,z,shares,y
            event('client_completed',client=cid,**peak())
        result['per_client_seconds']=[dict(mask=a,share=b,encode=c) for a,b,c in times]
        result['seconds']={n:sum(t[i] for t in times) for i,n in enumerate(['mask','share','encode'])}
        result['correctness']={'masking_relation_all_20d':True}
    elif args.role=='pipeline':
        event('preparation_begin')
        stored=[];prep=time.perf_counter()
        for cid in range(20):
            x,z,shares,y,ts=prepare(cid)
            selected=tuple(shares[p-1] for p in (4,1,3))
            stored.append((selected,z,y))
            del shares,x
        result['preparation_seconds']=time.perf_counter()-prep

        coeff=ref.lagrange_coefficients_zero((4,1,3),q)
        env.update(y_aggregate=[0]*d,aggregate_designated_shares={h:[0]*d for h in proto.verification_ids},
                   share_points=(4,1,3))
        event('kernels_begin',**peak());start=time.perf_counter()
        cv=sa=ma=checks=0.0
        for cid,(rows,z,y) in enumerate(stored):
            a=time.perf_counter()
            converted=[]
            for h,point,row,coefficient in zip(proto.verification_ids,(4,1,3),rows,coeff):
                packet=ref.PrioPacket(h,point,row,None,None,b'')
                env.update(packet=packet,coefficient=coefficient)
                exec(convert_code,env);converted.append(env['designated_additive'])
            cv+=time.perf_counter()-a
            a=time.perf_counter()
            assert all(sum(v[j] for v in converted)%q==z[j]%q for j in range(d))
            assert all(rows[k] is stored[cid][0][k] for k in range(3))
            checks+=time.perf_counter()-a
            a=time.perf_counter();env['submission']=type('RawComponentInput',(),{'gamma':type('PublicValue',(),{'y':y})()})()
            stmt('y_aggregate');ma+=time.perf_counter()-a
            a=time.perf_counter()
            for committee_id,original_share in zip(proto.verification_ids,rows):
                env.update(committee_id=committee_id,original_share=original_share);stmt('shareagg')
            sa+=time.perf_counter()-a
        a=time.perf_counter();stmt('z_aggregate');recon=time.perf_counter()-a
        a=time.perf_counter();stmt('aggregate_mask');center_z=time.perf_counter()-a
        a=time.perf_counter();stmt('recovered');stmt('centered');unmask=time.perf_counter()-a
        pipeline=time.perf_counter()-start
        event('kernels_complete',**peak())
        expected=[sum((j+7*i)%31-15 for i in range(20)) for j in range(d)]
        assert env['centered']==expected
        assert all(env['recovered'][j]==expected[j]%M for j in range(d))

        assert all(env['aggregate_mask'][j]==sum(record[1][j] for record in stored) for j in range(d))
        if args.write_vectors:
            target=args.output_dir/f'{args.profile}_pipeline_d{d}_r{args.repeat}'
            for suffix,values_ in [('expected',expected),('actual',env['centered'])]:
                vector_path=target.with_suffix('.'+suffix+'.i32')
                if vector_path.exists():
                    raise FileExistsError(f'Refusing to overwrite {vector_path}')
                vector_path.write_bytes(array.array('i',values_).tobytes())
        result['seconds']=dict(convert=cv,shareagg=sa,maskedagg=ma,recon=recon,center_mask=center_z,
                               unmask=unmask,pipeline=pipeline,inline_conversion_checks=checks)
        result['pipeline_boundary']='Continuous conversion+two aggregations+reconstruction+center/unmask, including inline conversion checks and loop overhead; excludes preparation and independent final oracles. Never add pipeline to children.'
        result['correctness']={'conversion_all_20d':True,'same_retained_raw_rows':True,
                              'aggregate_mask_exact':True,'aggregate_integer_exact':True,'aggregate_mod_M_exact':True}
    elif args.role=='constraint':

        x,z,shares,y,_=prepare(0)
        witness=ref.VMaskWitnessValues(tuple(x),tuple(map(tuple,env['limbs'])),
            tuple(tuple(v+cfg.r0 for v in row) for row in env['limbs']),tuple(env['delta']),tuple(z))
        statement=proto._formal_statement(sid='component-constraint',client_id=0,y=y)
        a=time.perf_counter();circuit,layout=ref.build_vmask_circuit(cfg,statement);build=time.perf_counter()-a
        a=time.perf_counter();expanded=ref.expand_witness(cfg,statement,witness);expand=time.perf_counter()-a
        a=time.perf_counter();ev=circuit.evaluate(expanded);valid=time.perf_counter()-a
        assert not any(ev.constraints)
        valid_count=len(ev.constraints);del ev
        malformed=list(expanded);malformed[layout.z.start]=(malformed[layout.z.start]+1)%q
        a=time.perf_counter();bad=circuit.evaluate(malformed);invalid=time.perf_counter()-a
        assert any(bad.constraints)
        result['seconds']=dict(circuit_build=build,witness_expand=expand,constraint_valid=valid,constraint_invalid=invalid)
        result['actual_processed_input_coordinates']=d
        result['correctness']={'valid_zero_constraints':True,'malformed_z_plus_one_nonzero':True,
                               'constraints':valid_count,'multiplication_gates':circuit.multiplication_count}
    result.update(status='PASS',memory=peak(),memory_scope='OS high-water RSS of this independent role process, including imports, preparation and allocator retention.')
    event('complete',role=args.role,d=d,seconds=result['seconds'],**peak())
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--role',choices=['client','pipeline','constraint'],required=True)
    ap.add_argument('--profile',choices=['P80','P100'],default='P80')
    ap.add_argument('--d',type=int,required=True);ap.add_argument('--repeat',type=int,default=0)
    ap.add_argument('--limit-mib',type=int,default=0,
                    help='optional Windows private-commit limit; 0 disables it')
    ap.add_argument('--write-vectors',action='store_true',
                    help='also retain expected and actual int32 output vectors')
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args()
    if a.d <= 0 or a.repeat < 0:
        ap.error('--d must be positive and --repeat must be nonnegative')
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/f'{a.profile}_{a.role}_d{a.d}_r{a.repeat}.json'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    try:
        result=run(a)
    except BaseException as exc:
        traceback.print_exc()
        result={'status':'FAIL','role':a.role,'profile':a.profile,'d':a.d,'repeat':a.repeat,
                'exception':repr(exc),'traceback':traceback.format_exc()}
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    sys.exit(0 if result['status']=='PASS' else 1)
