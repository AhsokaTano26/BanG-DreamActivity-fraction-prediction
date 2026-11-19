"""Sequential V3 conditional-calibration experiment over sealed blind caches."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform,shutil,tempfile
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.current_benchmark import load_dataset
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.v3_next_runner import DATASET_ID,FEATURE_COLUMNS
from analysis import v3_coupled_models as models

SOURCE=ROOT/'analysis/rolling_results/v3_next_warm_20261116'
SOURCE_ID='1a2aea3070785aaa377817f105239961b6b8582749ffe6ba3369fe6ffe81a9a7'
RUN=ROOT/'analysis/rolling_results/v3_coupled_adaptation_20261117'
SPEC=ROOT/'docs/superpowers/specs/2026-11-17-coupled-adaptation-design.md'
CANDIDATES=['uncorrected','ewma_reference',*models.RECIPES]
METHODS=CANDIDATES+[f'context_gate_seed{s}' for s in models.SEEDS]
REFERENCES={'kernel':'mape_kernel','gap':'coupled_gap','warm':'ewma_warm'}

def configuration(data,source,allow_fixture):
    seal=verify_seal(source/'blind_manifest.json');cfg=json.loads((source/'run_manifest.json').read_text())
    if (seal['run_id']!=cfg['run_id'] or identity({k:v for k,v in cfg.items() if k!='run_id'})!=cfg['run_id']
            or seal['dataset_id']!=data.manifest['dataset_id'] or cfg.get('experiment')!='warm_prequential_residual_optimization'):
        raise ValueError('Wrong sealed warm source configuration')
    if not allow_fixture and (data.manifest['dataset_id']!=DATASET_ID or cfg['run_id']!=SOURCE_ID or cfg['allow_fixture']):
        raise ValueError('Formal run requires exact denseV3 warm source')
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest or sha256(source/'sources'/name)!=digest:raise ValueError('Cached source changed')
    train=sorted([r for r in data.event_index.values() if r['split']=='train'],key=lambda r:(r['start_ms'],r['event']))
    if cfg['warmup_events']+cfg['warm_forecast_events']!=[r['event'] for r in train]:raise ValueError('Wrong warm event chronology')
    sources=local_sources([Path(__file__)])
    body={'protocol':'rolling_residual_v3','experiment':'conditional_coupled_adaptation','dataset_id':data.manifest['dataset_id'],
        'dataset_directory':str(data.root.resolve()),'base_run':str(source.resolve()),'base_run_id':cfg['run_id'],
        'base_blind_sha256':sha256(source/'blind_manifest.json'),'source_cache_specification_sha256':cfg['specification_sha256'],
        'warmup_events':cfg['warmup_events'],'warm_forecast_events':cfg['warm_forecast_events'],'methods':METHODS,
        'candidates':CANDIDATES,'recipes':models.RECIPES,'scored_splits':['validation','test'],
        'selection':'minimum validation event-macro MAPE; fixed candidate order ties; seed outputs never eligible',
        'training_scope':'cached base initial warmup, then sealed prequential calibration forecasts; no fit residuals',
        'feedback':'strictly completed already sealed own forecasts only, start-frozen history for all event prefixes',
        'resource_policy':{'processes':1,'threads':1,'nice':19,'background':True},'allow_fixture':bool(allow_fixture),
        'exposure':'Previously explored test, repeatedly used validation; exploratory follow-up only',
        'metadata_caveat':data.manifest['protocol']['metadata_caveat'],'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','catboost','joblib']},
        'source_hashes':{n:sha256(ROOT/n) for n in sources},'specification_sha256':sha256(SPEC)}
    body['run_id']=identity(body)
    return body,cfg

def check_cache_event(data,row,cfg,seal,audit,peer):
    if any(seal.get(k)!=v for k,v in {'run_id':cfg['run_id'],'event':row['event'],'split':row['split'],
          'query_start_ms':row['start_ms'],'end_ms':row['end_ms']}.items()):raise ValueError('Wrong cache identity/clock')
    if audit['current_outcome_released'] is not False or audit['query_start_ms']!=row['start_ms']:raise ValueError('Cache current outcome or clock invalid')
    expected={r['event']:r for r in data.event_index.values() if r['end_ms']<row['start_ms']}
    released=audit['released_history']
    if len(released)!=len(expected) or {r['event'] for r in released}!=set(expected):raise ValueError('Wrong cache base history')
    for r in released:
        if any(r[k]!=expected[r['event']][k] for k in ['start_ms','end_ms','sha256']):raise ValueError('Cache history date/hash changed')
    queries=set(cfg['warm_forecast_events'])|{r['event'] for r in data.event_index.values() if r['split']!='train'}
    if {r['event'] for r in audit['released_forecasts']}!=set(expected)&queries:raise ValueError('Wrong cached residual history')
    available=peer.query('peer_available==1')
    if (available.peer_observed_at>available.observed_at).any():raise ValueError('Peer suffix exceeds own observation')
    if (peer.observed_at>peer.cutoff).any():raise ValueError('Own observation exceeds cutoff')

def run_experiment(data,run,*,source_run=SOURCE,allow_fixture=False,resume=False):
    run,source=Path(run),Path(source_run)
    cfg,source_cfg=configuration(data,source,allow_fixture)
    if run.exists():
        if not resume:raise FileExistsError('Use a new run directory')
        if json.loads((run/'run_manifest.json').read_text())!=cfg:raise ValueError('Changed run cannot resume')
        if not (run/'blind_manifest.json').exists():raise FileExistsError('Incomplete run preserved; use new directory')
        verify_seal(run/'blind_manifest.json');return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True);(run/'events').mkdir();write_new_json(run/'run_manifest.json',cfg)
    for name in cfg['source_hashes']:
        dst=run/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    events=sorted(data.event_index.values(),key=lambda r:(r['start_ms'],r['event']))
    queries=[r for r in events if r['event'] in cfg['warm_forecast_events'] or r['split']!='train']
    expected_samples=defaultdict(list)
    for split in ['train','validation','test']:
        for sample in data.inputs(split):expected_samples[sample['curve'].event].append(sample)
    history=None;released={};predictions={}

    def release_before(start):
        nonlocal history
        for row in sorted(queries,key=lambda r:(r['end_ms'],r['start_ms'],r['event'])):
            event=row['event']
            if row['end_ms']>=start or event in released:continue
            path=run/'events'/str(event);sealed=verify_seal(path/'event_manifest.json')
            if sealed['run_id']!=cfg['run_id'] or sealed['event']!=event:raise ValueError('Feedback seal identity changed')
            curves=data.reveal_event(event,before_ms=start);finals={c.rank:c.final for c in curves}
            frame=pd.read_csv(path/'features.csv')
            frame['log_error']=[float(np.log(r.base_prediction)-np.log(finals[int(r.rank)])) for r in frame.itertuples()]
            history=frame if history is None or history.empty else pd.concat([history,frame],ignore_index=True)
            released[event]={k:row[k] for k in ['event','split','start_ms','end_ms','sha256']}
            released[event]['prediction_seal_sha256']=sha256(path/'event_manifest.json')

    def predict_event(row):
        nonlocal history
        event=row['event'];release_before(row['start_ms']);cached=source/'events'/str(event)
        src_seal=verify_seal(cached/'event_manifest.json');audit=json.loads((cached/'audit.json').read_text());peer=pd.read_csv(cached/'peer_audit.csv')
        check_cache_event(data,row,source_cfg,src_seal,audit,peer)
        frame=pd.read_csv(cached/'features.csv')[FEATURE_COLUMNS].copy()
        samples=expected_samples[event];ids=[s['case_id'] for s in samples]
        if frame.case_id.duplicated().any() or set(frame.case_id)!=set(ids) or set(peer.case_id)!=set(ids):raise ValueError('Cache case panel differs')
        frame=frame.set_index('case_id',drop=False).loc[ids].reset_index(drop=True)
        for r,s in zip(frame.itertuples(),samples):
            if (r.rank!=s['curve'].rank or r.event!=event or r.start_ms!=row['start_ms'] or r.end_ms!=row['end_ms']
                or r.cutoff!=s['cutoff'] or r.observed_at!=s['now'] or not np.isclose(r.current,s['score'],rtol=1e-12,atol=1e-6)):
                raise ValueError('Cache feature values differ from blind prefixes')
        saved=pd.read_csv(cached/'predictions.csv');refs={}
        for name,method in REFERENCES.items():
            p=saved.query('method==@method').set_index('case_id').loc[ids]
            refs[name]=p.predicted.to_numpy()
        raw=saved.query("method=='uncorrected'").set_index('case_id').loc[ids].predicted.to_numpy()
        if not np.allclose(raw,frame.base_prediction,rtol=1e-12,atol=1e-6):raise ValueError('Raw base cache differs')
        cold=saved.query("method=='ewma_reference'").set_index('case_id').loc[ids].predicted.to_numpy()
        if history is None:
            history=frame.iloc[:0].copy();history['log_error']=pd.Series(dtype=float)
        temporary=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        scores,features,details=models.predict_all(history.copy(deep=True),frame,refs,query_start_ms=row['start_ms'],model_directory=temporary/'models')
        scores={'uncorrected':raw,'ewma_reference':cold,**scores}
        if set(scores)!=set(METHODS):raise ValueError('Candidate menu changed')
        output=[]
        for method,values in scores.items():
            for case,value in zip(frame.to_dict('records'),values):
                output.append({k:case[k] for k in ['case_id','event','rank','phase','current']}|{
                    'method':method,'predicted':float(value),'log_bias':float(np.log(case['base_prediction']/value))})
        pd.DataFrame(output).to_csv(temporary/'predictions.csv',index=False);features.to_csv(temporary/'features.csv',index=False)
        history.to_csv(temporary/'released_residuals.csv',index=False);peer.to_csv(temporary/'peer_audit.csv',index=False)
        write_new_json(temporary/'audit.json',{'event':event,'split':row['split'],'query_start_ms':row['start_ms'],
            'current_outcome_released':False,'source_event_manifest_sha256':sha256(cached/'event_manifest.json'),
            'source_base_history':audit['released_history'],'released_forecasts':list(released.values()),
            'selection_sha256':sha256(run/'selection.json') if row['split']=='test' else None})
        seal_directory(temporary,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],
            'query_start_ms':row['start_ms'],'end_ms':row['end_ms'],'cases':len(frame),'methods':METHODS})
        temporary.rename(run/'events'/str(event))
        print(json.dumps({'sealed_event':event,'split':row['split'],'history_events':len(released),'cases':len(frame),'methods':len(scores)}),flush=True)
        return pd.DataFrame(output)

    for split in ['train','validation','test']:
        rows=[r for r in queries if r['split']==split]
        if split=='test':
            release_before(rows[0]['start_ms']);truth=history.set_index('case_id');values={}
            for method in CANDIDATES:
                part=predictions['validation'].query('method==@method');actual=truth.loc[part.case_id]
                real=np.exp(np.log(actual.base_prediction.to_numpy())-actual.log_error.to_numpy())
                errors=pd.DataFrame({'event':part.event.to_numpy(),'ape':np.abs(part.predicted.to_numpy()/real-1)*100})
                values[method]=float(errors.groupby('event').ape.mean().mean())
            selected=min(CANDIDATES,key=lambda k:(values[k],CANDIDATES.index(k)))
            write_new_json(run/'selection.json',{'run_id':cfg['run_id'],'selected_recipe':selected,'methods':METHODS,'candidates':CANDIDATES,
                'validation_scores':values,'criterion':cfg['selection'],'test_labels_used_for_selection':False,'test_inputs_loaded_for_selection':False,
                'sealed_before_test_start_ms':rows[0]['start_ms'],'validation_event_seals':{str(r['event']):sha256(run/'events'/str(r['event'])/'event_manifest.json') for r in queries if r['split']=='validation'}})
        predictions[split]=pd.concat([predict_event(row) for row in rows],ignore_index=True)
        expected={s['case_id'] for row in rows for s in expected_samples[row['event']]}
        for method,p in predictions[split].groupby('method'):
            if p.case_id.duplicated().any() or set(p.case_id)!=expected:raise ValueError('Frozen case coverage changed')
        predictions[split].to_csv(run/('warm_predictions.csv' if split=='train' else f'{split}_predictions.csv'),index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during run')
    seal_directory(run,'blind_manifest.json',{'protocol':'rolling_residual_v3','experiment':cfg['experiment'],'run_id':cfg['run_id'],
        'dataset_id':cfg['dataset_id'],'dataset_directory':cfg['dataset_directory'],'methods':METHODS,'scored_splits':['validation','test'],
        'reference_method':'ewma_reference','selected_recipe':selected,'base_blind_sha256':cfg['base_blind_sha256']})
    return json.loads((run/'selection.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--source-run',type=Path,default=SOURCE);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(run_experiment(load_dataset('v3'),a.run,source_run=a.source_run,resume=a.resume),ensure_ascii=False))
