"""Independent source-to-number inspection; never import either G binder."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,math,re,statistics,zipfile
from functools import lru_cache
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(None)
def load(p):
    p=Path(p)
    if p.suffix=='.json':return json.loads(p.read_text())
    with p.open(newline='') as f:return list(csv.DictReader(f))
def access(obj,field):
    for key in re.findall(r'[^.\[\]]+',field):obj=obj[int(key)] if isinstance(obj,list) else obj[key]
    return obj
def objects(obj):
    if isinstance(obj,dict):
        yield obj
        for x in obj.values():yield from objects(x)
    elif isinstance(obj,list):
        for x in obj:yield from objects(x)
def equal(a,b):
    try:return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
    except (TypeError,ValueError):return a==b
def selected(data,where):return [x for x in data if all(str(x.get(k))==str(v) for k,v in where.items())]
def run(paper):
    report={'scope':'All source hashes, source selectors, scalar/aggregate arithmetic, rendered rounding and baseline preservation. No G binder imported.','paper_root':str(paper),'manifests':{},'findings':[]}
    for filename,macrofile in [('upgrade_number_bindings.json','upgrade_macros.tex'),('round2_number_bindings.json','round2_macros.tex')]:
        path=paper/'qa'/filename;m=json.loads(path.read_text());start_sha=sha(path);errors=[];checked=[];unit_issues=[]
        for rel,digest in m['sources'].items():
            if sha(ROOT/rel)!=digest:errors.append({'check':'source_sha256','path':rel})
        for rel,digest in m['generated_files'].items():
            if sha(paper/rel)!=digest:errors.append({'check':'generated_sha256','path':rel})
        if sha(paper/'data'/macrofile)!=m['macros_sha256']:errors.append({'check':'macro_sha256'})
        text='\n'.join((paper/rel).read_text() for rel in m['generated_files'])+'\n'+(paper/'data'/macrofile).read_text();macros=dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}',(paper/'data'/macrofile).read_text()));unused_legacy=[];selector_context=[]
        for b in m['bindings']:
            try:
                name=b['binding'];p=ROOT/b['source'];data=load(str(p));where=b.get('row_condition',{});field=b.get('field');agg=b.get('aggregation');raw=None;special=False
                if sha(p)!=b['source_sha256']:raise ValueError('binding source SHA mismatch')
                if b.get('paired_source') and sha(ROOT/b['paired_source'])!=b['paired_sha256']:raise ValueError('paired source SHA mismatch')
                if isinstance(data,list):chosen=selected(data,where)
                elif where.get('object'):chosen=[data[where['object']]]
                elif where:
                    # Legacy summary contains both validation and test effects;
                    # its archived selector omits the parent JSON path. These
                    # manuscript entries explicitly describe the test panel.
                    if filename.startswith('upgrade') and p.name=='summary.json' and 'splits' in data:
                        chosen=selected(data['splits']['test']['paired_effects'],where);selector_context.append({'binding':name,'explicit_parent_path':'splits.test.paired_effects'})
                    else:chosen=[obj for obj in objects(data) if all(str(obj.get(k))==str(v) for k,v in where.items())]
                else:chosen=[data]
                if agg=='sum(cpu_seconds),sum(wall_seconds);unique(parameters)':
                    if not chosen:raise ValueError('empty aggregate')
                    cp=sum(float(x['cpu_seconds']) for x in chosen);wall=sum(float(x['wall_seconds']) for x in chosen);params=sorted({float(x['parameters']) for x in chosen if x.get('parameters') not in ['',None]})
                    if len(chosen)!=b['fits'] or not equal(cp,b['cpu_seconds']) or not equal(wall,b['wall_seconds']) or params!=b['parameters']:raise ValueError('budget aggregate mismatch')
                    special=True
                elif agg=='equal mean of all 16 cells x 4 history lengths':
                    if len(chosen)!=64:raise ValueError('expected64 synthetic cells')
                    raw=statistics.fmean(float(x[field]) for x in chosen)
                    paired=load(str(ROOT/b['paired_source']));direction={}
                    for key,metric in [('state','state_mse'),('ape','synthetic_unbounded_ape_pct')]:
                        z=[x for x in paired if x.get('method')==where['method'] and x.get('metric')==metric]
                        if not z and key=='ape':z=[x for x in paired if x.get('method')==where['method'] and 'ape' in x.get('metric','')]
                        if z:
                            col='mean' if 'mean' in z[0] else 'mean_difference_vs_independent'
                            if col not in z[0]:col=next(k for k in z[0] if 'diff' in k and not ('low' in k or 'high' in k))
                            direction[key]=[sum(float(x[col]) < -1e-12 for x in z),sum(float(x[col]) > 1e-12 for x in z)]
                    if direction and direction!=b['direction_counts']:raise ValueError('synthetic direction counts mismatch: '+str(direction))
                elif agg=='negative/positive count of 24 fixed regime means':
                    if len(chosen)!=24:raise ValueError('expected24 simulation regimes')
                    raw=[sum(float(x[field])<0 for x in chosen),sum(float(x[field])>0 for x in chosen)]
                elif agg=='R_raw_00 / R_raw_11':
                    if len(chosen)!=1:raise ValueError('ratio selector is not unique')
                    raw=float(chosen[0][field[0]])/float(chosen[0][field[1]])
                elif field:
                    if len(chosen)!=1:raise ValueError('selector is not unique: '+str(len(chosen)))
                    raw=access(chosen[0],field)
                    if isinstance(raw,dict) and b.get('dataset_id'):
                        if raw['dataset_id']!=b['dataset_id'] or raw['cases']!=b['cases']:raise ValueError('dataset metadata mismatch')
                        special=True
                else:raise ValueError('unhandled binding schema')
                if not special:
                    value=raw if isinstance(raw,list) else float(raw)*b.get('multiplier',1)
                    if isinstance(value,list):
                        if value!=b['value']:raise ValueError('aggregate value mismatch')
                        rendered='/'.join(map(str,value))
                    else:
                        if not equal(value,b['value']):raise ValueError('numeric value mismatch')
                        if b.get('conversion'):
                            rendered=datetime.datetime.fromtimestamp(value/1000,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]+' UTC'
                        else:
                            match=re.search(r'(\d+) decimal',b.get('rounding',''))
                            if match:digits=int(match.group(1))
                            else:digits=len(b['rendered'].split('.')[1]) if '.' in b['rendered'] else 0
                            rendered=f'{value:.{digits}f}'
                    if rendered!=b['rendered']:raise ValueError('rendered rounding mismatch: '+rendered)
                    if 'raw_value' in b and not agg and not equal(raw,b['raw_value']):raise ValueError('raw field mismatch')
                    if rendered not in text and not b.get('conversion'):
                        if filename.startswith('upgrade'):
                            unused_legacy.append(name)
                        else:raise ValueError('rendered value missing from generated text')
                    if name in macros and macros[name]!=rendered:raise ValueError('named macro differs from binding')
                checked.append(name)
                # Unit identity is separate from numerical matching.
                if filename.startswith('round2'):
                    unit=b.get('unit','').lower();f=str(field)
                    if ('lambda' in f or f in ['task_states','lambda_positive_states']) and 'percentage' in unit:unit_issues.append({'binding':name,'field':field,'unit':b.get('unit')})
                    if 'count or seconds' in unit or ' or ' in unit:unit_issues.append({'binding':name,'field':field,'unit':b.get('unit')})
                    if ('mape' in f.lower() and 'delta' not in f.lower()) and 'percentage points' in unit:unit_issues.append({'binding':name,'field':field,'unit':b.get('unit')})
            except Exception as exc:errors.append({'binding':b.get('binding'),'error':str(exc)})
        stable=sha(path)==start_sha
        report['manifests'][filename]={'manifest_sha256':start_sha,'stable_during_audit':stable,'bindings':len(m['bindings']),'checked_bindings':len(checked),'sources':len(m['sources']),'tables':len(m['generated_files']),'macros':len(macros),'errors':errors,'unit_issues':unit_issues,'generated_file_shas':m['generated_files'],'unused_valid_legacy_entries':unused_legacy,'legacy_selector_parent_context':selector_context}
    # Old native outputs and every numeric/aggregate entry must remain baseline.
    with zipfile.ZipFile(BASE/'baseline/round1_complete_manuscript.zip') as z:
        prefix='papers/20261118_entertainment_computing_cn/';old=json.loads(z.read(prefix+'qa/upgrade_number_bindings.json'));current=json.loads((paper/'qa/upgrade_number_bindings.json').read_text());numbers=[];outputs=[]
        for a,b in zip(old['bindings'],current['bindings']):
            ignore={'source','source_sha256'}
            if {k:v for k,v in a.items() if k not in ignore}!={k:v for k,v in b.items() if k not in ignore}:numbers.append(a['binding'])
        for rel in list(old['generated_files'])+['data/upgrade_macros.tex']:
            if z.read(prefix+rel)!=(paper/rel).read_bytes():outputs.append(rel)
        report['baseline_preservation']={'same_binding_count':len(old['bindings'])==len(current['bindings']),'changed_numeric_entries':numbers,'changed_native_outputs':outputs}
    report['status']='passed' if all(not d['errors'] and not d['unit_issues'] and d['stable_during_audit'] for d in report['manifests'].values()) and not report['baseline_preservation']['changed_numeric_entries'] and not report['baseline_preservation']['changed_native_outputs'] else 'review'
    report['script_sha256']=sha(__file__);(HERE/'H_final_binding_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'manifests':{k:{x:d[x] for x in ['bindings','checked_bindings','tables','macros','errors']}|{'unit_issues':len(d['unit_issues'])} for k,d in report['manifests'].items()},'baseline':report['baseline_preservation']},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper-root',type=Path,default=BASE/'workspaces/G/manuscript');run(p.parse_args().paper_root.resolve())
