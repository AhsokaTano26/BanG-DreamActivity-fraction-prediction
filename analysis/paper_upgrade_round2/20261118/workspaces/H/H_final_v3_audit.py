"""External final sidecar: v3 CRC/member/core parity and slim-source native build."""
import hashlib,json,os,re,signal,stat,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2];PAPER=ROOT/'papers/20261118_entertainment_computing_cn';PREFIX='papers/20261118_entertainment_computing_cn/';START=time.monotonic();DEADLINE=START+30
OUT=HERE/'H_final_v3_sidecar';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected={'round2_latex_source_v3.zip':'d947f52a8b051e2054a32dc52a8d7f5d38d0b4319c918c91f400d54d805c24c5','round2_research_bundle_v3.zip':'70d03465dc02f08f5635457e469d661efba4c12e5046ef0dd84eba68a2bc4590','round2_figures_v3.zip':'13be46b0b5662199ffbf620dea4052045eed73c035e39a9c4f5565908955a9d3'}
receipt=json.loads((BASE/'delivery/v3/receipt.json').read_text());temp=Path(tempfile.mkdtemp(prefix='round2_H_final_source_v3_'));source_project=temp/'project';source_project.mkdir();reports={};payload={};documents={}
for archive in receipt['archives']:
    path=ROOT/archive['file'];bad=[];unsafe=[]
    with zipfile.ZipFile(path) as z:
        names=z.namelist();members=json.loads(z.read('DELIVERY_MANIFEST.json'))['members']
        for info in z.infolist():
            rel=PurePosixPath(info.filename);mode=(info.external_attr>>16)&0o170000
            if rel.is_absolute() or '..' in rel.parts or '\\' in info.filename or mode==stat.S_IFLNK:unsafe.append(info.filename)
        crc=z.testzip()
        for rel,meta in members.items():
            content=z.read(rel)
            if hashlib.sha256(content).hexdigest()!=meta['sha256'] or len(content)!=meta['bytes']:bad.append(rel)
        valid=sha(path)==archive['sha256']==expected[path.name] and not bad and not unsafe and crc is None and len(names)==len(set(names))==len(members)+1 and set(names)==set(members)|{'DELIVERY_MANIFEST.json'}
        reports[path.name]={'passed':valid,'ZIP_sha256':sha(path),'members':len(names),'checked_payload':len(members),'bytes':path.stat().st_size,'bad_members':bad,'unsafe_members':unsafe,'CRC_bad_member':crc};payload[path.name]=members
        if path.name=='round2_latex_source_v3.zip':
            if not valid:raise ValueError('Source integrity failed')
            z.extractall(source_project)
        if path.name=='round2_research_bundle_v3.zip':
            for rel in ['delivery_manifest.json','delivery_receipt.json']:documents[rel]=json.loads(z.read(PREFIX+rel))
    print(json.dumps({'archive':path.name,**reports[path.name]}),flush=True)
    if time.monotonic()>DEADLINE:raise TimeoutError('30second budget reached during CRC/hash audit')
with zipfile.ZipFile(PAPER/'output/round2_research_bundle_v1.zip') as z:v1=json.loads(z.read('DELIVERY_MANIFEST.json'))['members']
v3=payload['round2_research_bundle_v3.zip'];source=payload['round2_latex_source_v3.zip'];checks={};changes=[name for name in v1.keys()&v3.keys() if v1[name]!=v3[name]]
def core(name):
    return name.endswith(('.pt','.joblib','.tex','.bib','.svg','.png','.pdf','.csv','.csv.gz','.jsonl','.jsonl.gz')) or (name.endswith('.py') and not name.endswith('/integration/package_round2.py')) or name.endswith('/requirements-upgrade.txt') or name=='AGENTS.md'
bad_core=[name for name in v1 if core(name) and (name not in v3 or v1[name]!=v3[name])]
checks['three_archives_CRC_SHA_and_safe_members']=all(v['passed'] for v in reports.values());checks['all_v1_scientific_core_unchanged']=not bad_core
driver='analysis/paper_upgrade_round2/20261118/integration/reproduce_paper.py';checks['one_command_driver_unchanged']=v1[driver]==v3[driver]
checks['legacy16_metadata_preserved_with_explicit_archive']=all(v1[PREFIX+rel]==v3[PREFIX+'legacy_delivery_metadata/'+rel] for rel in ['delivery_manifest.json','delivery_receipt.json'])
manifest=documents['delivery_manifest.json'];current=documents['delivery_receipt.json'];bad_nested=[path for path,h in manifest['source_files'].items() if v3.get(PREFIX+path,{}).get('sha256')!=h]
checks['current_delivery_identity_and_nested_SHA']=(manifest['main_pages_including_supplement']==45 and manifest['standalone_supplement_pages']==19 and manifest['unique_figures']==18 and manifest['references_main']==33 and manifest['tests_repository']+manifest['tests_new']==539 and current['version']=='round2_final_v3' and current['counts']=={'main_pages':45,'standalone_supplement_pages':19,'figures':18,'references_main':33,'unique_tests':539} and current['future_events']==0 and current['future_accuracy'] is None and not bad_nested)
checks['source_payload_matches_research']=all(v3.get(name)==data for name,data in source.items());checks['source_is_slim_native_payload']=not any(name.endswith(('.svg','.png')) for name in source) and sum(name.startswith(PREFIX+'figures/') and name.endswith('.pdf') for name in source)==18 and reports['round2_latex_source_v3.zip']['bytes']<2000000
paper=source_project/'papers/20261118_entertainment_computing_cn';latex='/Library/TeX/texbin/xelatex';bib='/Library/TeX/texbin/bibtex';command=['make','all','PYTHON='+sys.executable,'REPO_ROOT='+str(source_project),'LATEX='+latex,'BIBTEX='+bib];env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
print(json.dumps({'source_only_native_command':command,'remaining_seconds':DEADLINE-time.monotonic()}),flush=True)
with (OUT/'source_make_all.log').open('w') as log:
    process=subprocess.Popen(command,cwd=paper,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=max(1,DEADLINE-time.monotonic()))
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait();code=-1
checks['slim_source_native_make_all']=code==0
pdfs={};warnings=[]
if code==0:
    for kind,count in [('main',45),('supplement',19)]:
        a=subprocess.check_output(['pdftotext','-layout',str(PAPER/'output'/f'{kind}.pdf'),'-'],text=True);b=subprocess.check_output(['pdftotext','-layout',str(paper/'output'/f'{kind}.pdf'),'-'],text=True);info=subprocess.check_output(['pdfinfo',str(paper/'output'/f'{kind}.pdf')],text=True);fonts=subprocess.check_output(['pdffonts',str(paper/'output'/f'{kind}.pdf')],text=True);log=(paper/'output'/f'{kind}.log').read_text(errors='replace');badfont=[line for line in fonts.splitlines()[2:] if not re.search(r'\s+yes\s+(?:yes|no)\s+yes\s+\d+\s+\d+\s*$',line)];badwarnings=[line for line in log.splitlines() if any(s in line for s in ['Overfull \\hbox','Overfull \\vbox','undefined','Missing character:','multiply defined'])]
        pdfs[kind]={'text_equal':a==b,'pages':int(re.search(r'Pages:\s+(\d+)',info).group(1)),'fonts_embedded_unicode':not badfont,'latex_warnings':badwarnings,'compiled_pdf_SHA':sha(paper/'output'/f'{kind}.pdf')};checks[kind+'_compiled_text_pages_fonts']=a==b and pdfs[kind]['pages']==count and not badfont and not badwarnings
else:warnings.append('Native compile failed or budget exceeded; logs retained')
result={'passed':all(checks.values()),'scope':'Final external sidecar; not included again in v3 archive. CRC/all memberSHA, v1 scientific core parity, nested metadata identity and actual standalone native source compilation.','checks':checks,'archives':reports,'common_member_changes':changes,'scientific_core_changes':bad_core,'bad_current_nested_SHA':bad_nested,'PDFs':pdfs,'elapsed_seconds':time.monotonic()-START,'budget_seconds':30,'temporary_source_project':str(source_project),'no_full_pipeline_or_model_replay':True,'script_sha256':sha(__file__),'failures':warnings,'version_history':'v1 full external pipeline accepted; v2 slim package retained as nonfinal; v3 archives legacy16 metadata and is the final delivery candidate.'}
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'elapsed_seconds':result['elapsed_seconds'],'checks':checks,'changed_common_members':changes,'result':str(OUT/'result.json')},ensure_ascii=False))
