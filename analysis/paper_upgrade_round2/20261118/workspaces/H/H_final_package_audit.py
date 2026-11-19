"""Final package integrity and external one-command build; no fitting."""
from __future__ import annotations
import hashlib,json,os,signal,stat,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2];PAPER=ROOT/'papers/20261118_entertainment_computing_cn';START=time.monotonic();DEADLINE=START+180
OUT=HERE/'H_final_package_v1';OUT.mkdir(exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest_bytes(v):return hashlib.sha256(v).hexdigest()
receipt=json.loads((BASE/'delivery/v1/receipt.json').read_text());reports={};payloads={}
expected={'round2_research_bundle_v1.zip':'1a83f8c79f94dc8b41234ad560781a4e299c449836e5df9755bdc37b3f88e1e7','round2_latex_source_v1.zip':'34b31aec9a18632f2b0414a4c61912548c91b850de76595b00d44695786a23c6','round2_figures_v1.zip':'decae98556dfa6cfe09cdb653750f1fc1be18c086eb053c7538b5e00484aa3f5'}
temp=Path(tempfile.mkdtemp(prefix='round2_H_final_zip_'));project=temp/'project';project.mkdir()
for archive in receipt['archives']:
    path=ROOT/archive['file'];name=path.name;bad=[];unsafe=[]
    with zipfile.ZipFile(path) as z:
        names=z.namelist();members=json.loads(z.read('DELIVERY_MANIFEST.json'))['members']
        for info in z.infolist():
            rel=PurePosixPath(info.filename);mode=(info.external_attr>>16)&0o170000
            if rel.is_absolute() or '..' in rel.parts or '\\' in info.filename or mode==stat.S_IFLNK:unsafe.append(info.filename)
        crc=z.testzip()
        for rel,meta in members.items():
            with z.open(rel) as f:
                h=hashlib.sha256();size=0
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
            if h.hexdigest()!=meta['sha256'] or size!=meta['bytes']:bad.append(rel)
        valid=(not unsafe and not bad and crc is None and len(names)==len(set(names))==len(members)+1 and set(names)==set(members)|{'DELIVERY_MANIFEST.json'} and sha(path)==archive['sha256']==expected[name])
        reports[name]={'passed':valid,'ZIP_sha256':sha(path),'bytes':path.stat().st_size,'members':len(names),'member_sha_checked':len(members),'bad_members':bad,'unsafe_members':unsafe,'CRC_bad_member':crc}
        payloads[name]=members
        if name=='round2_research_bundle_v1.zip':
            if not valid:raise ValueError('Research package integrity failed')
            z.extractall(project)
    print(json.dumps({'archive':name,**reports[name]}),flush=True)
    if time.monotonic()>DEADLINE:raise TimeoutError('180-second budget reached during package checks')

research=payloads['round2_research_bundle_v1.zip'];source=payloads['round2_latex_source_v1.zip'];figures=payloads['round2_figures_v1.zip']
source_bad=[name for name,data in source.items() if name not in research or research[name]!=data];fig_bad=[name for name,data in figures.items() if name not in research or research[name]!=data]
closure=['analysis/backtest.py','analysis/event_features.py','analysis/holdout_dataset.py','analysis/holdout_evaluate.py','analysis/multirank_dataset.py','analysis/multirank_model.py','analysis/paper_upgrade_replay_models.py','analysis/rolling_residual_stats.py','download_ranks.py']
train=[f'data/temporal_holdout_v2_dense_t1000_t2000_20261115/{name}' for name in ['train_inputs.jsonl.gz','train_cases.csv','train_labels.csv']]
closure_report={name:{'present':name in research,'source_SHA_matches':(sha(project/name)==sha(ROOT/name)) if name in research else False} for name in closure+train}
(OUT/'archive_integrity.json').write_text(json.dumps({'archives':reports,'source_payload_not_in_research':source_bad,'figure_payload_not_in_research':fig_bad,'C_complete_closure_and_TRAIN':closure_report},indent=2)+'\n')
if source_bad or fig_bad or not all(v['present'] and v['source_SHA_matches'] for v in closure_report.values()):raise ValueError('Package closure/subset check failed')

# All Python children inherit the same read guard. Shared .venv dependencies and
# system fonts are allowed; the original project's source/data paths are denied.
guard=temp/'guard';guard.mkdir();guard_log=temp/'guard_events.jsonl'
site=r'''import hashlib,json,os,sys
original=os.path.realpath(os.environ['H_AUDIT_ORIGINAL_ROOT']);venv=os.path.realpath(os.path.join(original,'.venv'));log=os.environ['H_AUDIT_GUARD_LOG']
def record(event,path=None):
    with open(log,'a') as f:f.write(json.dumps({'event':event,'path':path,'pid':os.getpid()})+'\n')
record('hook_loaded')
def hook(event,args):
    if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=os.path.realpath(os.fsdecode(args[0]));under=path==original or path.startswith(original+os.sep);allowed=path==venv or path.startswith(venv+os.sep)
    if under and not allowed:
        record('denied_original_project_access',path);raise PermissionError('H original project read guard: '+path)
sys.addaudithook(hook)
'''
(guard/'sitecustomize.py').write_text(site)
env={**os.environ,'PYTHONPATH':str(guard),'PYTHONDONTWRITEBYTECODE':'1','H_AUDIT_ORIGINAL_ROOT':str(ROOT),'H_AUDIT_GUARD_LOG':str(guard_log),**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']}}
probe='from pathlib import Path\nimport os\ntry: Path(os.environ["H_AUDIT_ORIGINAL_ROOT"],"AGENTS.md").read_bytes()\nexcept PermissionError: print("guard-negative-control-passed")\nelse: raise RuntimeError("guard did not reject original source")\n'
negative=subprocess.run([sys.executable,'-B','-c',probe],cwd=project,env=env,capture_output=True,text=True,timeout=5,check=True)
(OUT/'guard_negative_control.log').write_text(negative.stdout+negative.stderr)
command=[sys.executable,'-B',str(project/'analysis/paper_upgrade_round2/20261118/integration/reproduce_paper.py'),'--repo-root',str(project)]
print(json.dumps({'starting_complete_one_command':command,'remaining_seconds':DEADLINE-time.monotonic()}),flush=True)
with (OUT/'one_command.log').open('w') as log:
    process=subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=max(1,DEADLINE-time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM);process.wait();code=-1
if code!=0:
    failure={'returncode':code,'temporary_project':str(project),'elapsed_seconds':time.monotonic()-START,'stage':'external one-command failed; logs preserved'}
    (OUT/'result.json').write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps(failure));raise SystemExit(1)

rebuilt=project/'papers/20261118_entertainment_computing_cn';checks={};checks['all_three_archives'] = all(v['passed'] for v in reports.values());checks['source_and_figure_archives_exact_payload_subsets']=not source_bad and not fig_bad
checks['C_closure_and_three_TRAIN_files']=all(v['present'] and v['source_SHA_matches'] for v in closure_report.values())
pixels=[]
for path in sorted((PAPER/'qa/round2_render').glob('*/page-*.png')):
    relative=path.relative_to(PAPER);pixels.append({'path':str(relative),'same_SHA':sha(path)==sha(rebuilt/relative)})
checks['all64_document_page_PNGs_equal']=len(pixels)==64 and all(v['same_SHA'] for v in pixels)
figpng=[]
for path in sorted((PAPER/'figures').glob('upgrade_*.png')):figpng.append({'name':path.name,'same_SHA':sha(path)==sha(rebuilt/'figures'/path.name)})
checks['all18_figure_PNGs_equal']=len(figpng)==18 and all(v['same_SHA'] for v in figpng)
native=[]
for directory,pattern in [('tables','upgrade_*.tex'),('tables','round2_*.tex'),('data','*_macros.tex')]:
    for path in sorted((PAPER/directory).glob(pattern)):native.append({'path':str(path.relative_to(PAPER)),'same_SHA':sha(path)==sha(rebuilt/path.relative_to(PAPER))})
checks['native_tables_macros_equal']=all(v['same_SHA'] for v in native)
documents={}
for kind,wanted in [('main',45),('supplement',19)]:
    a=subprocess.check_output(['pdftotext','-layout',str(PAPER/'output'/f'{kind}.pdf'),'-'],text=True);b=subprocess.check_output(['pdftotext','-layout',str(rebuilt/'output'/f'{kind}.pdf'),'-'],text=True);info=subprocess.check_output(['pdfinfo',str(rebuilt/'output'/f'{kind}.pdf')],text=True)
    documents[kind]={'text_equal':a==b,'pages':wanted,'pdfinfo':info,'rebuilt_pdf_SHA':sha(rebuilt/'output'/f'{kind}.pdf'),'canonical_pdf_SHA':sha(PAPER/'output'/f'{kind}.pdf')}
    checks[kind+'_text_and_pages']=a==b and f'Pages:           {wanted}' in info
events=[json.loads(line) for line in guard_log.read_text().splitlines()];denied=[v for v in events if v['event']=='denied_original_project_access'];checks['guard_negative_control_and_no_pipeline_original_reads']=len(denied)==1 and denied[0]['path']==str(ROOT/'AGENTS.md') and len({v['pid'] for v in events if v['event']=='hook_loaded'})>=3
newrender=json.loads((rebuilt/'qa/round2_render_manifest.json').read_text());checks['render_source_PDF_SHA_binding']=all(v['source_pdf_sha256']==sha(rebuilt/'output'/f'{kind}.pdf') for kind,v in newrender.items())
docchecks=json.loads((rebuilt/'qa/round2_document_checks.json').read_text());checks['package_native_verifier_passed']=docchecks['status']=='passed'
result={'passed':all(checks.values()),'checks':checks,'archives':reports,'temporary_project':str(project),'elapsed_seconds':time.monotonic()-START,'C_complete_closure_and_TRAIN':closure_report,'documents':documents,'figure_PNG_comparison':figpng,'page_PNG_comparison':pixels,'native_comparison':native,'hook_events':events,'external_source_guard_scope':'All Python pipeline processes; installed .venv and system fonts reused. External TeX tools use explicit extracted project paths; no fresh dependency install claim.','source_only_build_scope':'Source and figure archive payloads exactly match subsets of the successfully rebuilt research archive; no redundant separate source-only compile.','script_sha256':sha(__file__)}
(OUT/'guard_events.jsonl').write_text(guard_log.read_text());(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'elapsed_seconds':result['elapsed_seconds'],'checks':checks,'result':str(OUT/'result.json')},ensure_ascii=False))
