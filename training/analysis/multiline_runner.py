"""Guarded19rank calibration, sealed feedback and validation-only selection."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.multiline_dataset import MultilineDataset,DEFAULT
from analysis.multiline_base import RUN as BASE
from analysis import multiline_models as models
from download_ranks import RANKS

RUN=ROOT/'analysis/rolling_results/v3_multiline_calibration_20261117'
SPEC=ROOT/'docs/superpowers/specs/2026-11-17-multiline-v3-design.md'
CANDIDATES=list(models.RECIPES)
METHODS=CANDIDATES+[f'neural_gate_seed{s}' for s in models.SEEDS]
INPUT_COLUMNS=['case_id','event','rank','phase','progress','start_ms','end_ms','cutoff','observed_at','current',
               'event_type','band_group','split','duration_h','remaining_h','rate_6h_relative','rate_24h_relative']

def configuration(data,base,allow_fixture):
    seal=verify_seal(base/'blind_manifest.json');bcfg=json.loads((base/'run_manifest.json').read_text())
    if bcfg['run_id']!=seal['run_id'] or identity({k:v for k,v in bcfg.items() if k!='run_id'})!=bcfg['run_id'] or seal['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Wrong base cache identity')
    if not allow_fixture and (bcfg['allow_fixture'] or bcfg['base_recipe']!={'iterations':600,'depth':5,'learning_rate':.035,'l2_leaf_reg':12,'seed':42,'threads':1} or bcfg['ranks']!=RANKS):raise ValueError('Wrong formal base recipe')
    for name,digest in bcfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest or sha256(base/'sources'/name)!=digest:raise ValueError('Base source changed')
    sources=local_sources([Path(__file__)])
    cfg={'protocol':'rolling_multiline_calibration_v3','dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
        'base_run':str(base.resolve()),'base_run_id':bcfg['run_id'],'base_blind_sha256':sha256(base/'blind_manifest.json'),
        'context_run':bcfg['context_run'],'context_run_id':bcfg['context_run_id'],'ranks':bcfg['ranks'],
        'warmup_events':bcfg['warmup_events'],'warm_forecast_events':bcfg['warm_forecast_events'],
        'methods':METHODS,'candidates':CANDIDATES,'recipes':models.RECIPES,'reference_method':'ewma_reference',
        'selection':'minimum validation event-macro over all frozen19rank cases; fixed candidate order ties; no seed selection',
        'scored_splits':['validation','test'],'history':'start-frozen, strictly ended already sealed forecast events only',
        'training_scores':'35prequential activities after32base warmup; never resubstitution',
        'scope':'19rank totals and original2rank panels reported separately, not interchangeable',
        'resource_policy':{'processes':1,'threads':1,'nice':19,'background':True},'allow_fixture':bool(allow_fixture),
        'exposure':'Previously explored test, reused validation','metadata_caveat':data.manifest['protocol']['metadata_caveat'],
        'source_hashes':{n:sha256(ROOT/n) for n in sources},'specification_sha256':sha256(SPEC),'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','torch','catboost','joblib','scikit-learn']}}
    cfg['run_id']=identity(cfg)
    return cfg,bcfg

def run(data,run,*,base_run=BASE,allow_fixture=False,resume=False):
    run,base=Path(run),Path(base_run);cfg,bcfg=configuration(data,base,allow_fixture)
    if run.exists():
        if not resume:raise FileExistsError('New run directory required')
        if json.loads((run/'run_manifest.json').read_text())!=cfg or not (run/'blind_manifest.json').exists():raise ValueError('Only identical completed run can resume')
        verify_seal(run/'blind_manifest.json');return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True);(run/'events').mkdir();write_new_json(run/'run_manifest.json',cfg)
    for name in cfg['source_hashes']:
        p=run/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    queries=sorted([r for r in data.event_index.values() if r['event'] in cfg['warm_forecast_events'] or r['split']!='train'],key=lambda r:(r['start_ms'],r['event']))
    case_tables={s:data.case_index(s) for s in ['train','validation','test']}
    released={};history=None;predictions={}
    def release_before(start):
        nonlocal history
        for row in sorted(queries,key=lambda r:(r['end_ms'],r['start_ms'],r['event'])):
            event=row['event']
            if row['end_ms']>=start or event in released:continue
            path=run/'events'/str(event);seal=verify_seal(path/'event_manifest.json')
            if seal['event']!=event or seal['run_id']!=cfg['run_id']:raise ValueError('Own feedback seal mismatch')
            finals={c.rank:c.final for c in data.reveal_event(event,before_ms=start)}
            frame=pd.read_csv(path/'features.csv');frame['log_error']=[float(np.log(r.base_prediction)-np.log(finals[int(r.rank)])) for r in frame.itertuples()]
            history=frame if history is None or history.empty else pd.concat([history,frame],ignore_index=True)
            released[event]={k:row[k] for k in ['event','split','start_ms','end_ms','sha256']}
            released[event]['prediction_seal_sha256']=sha256(path/'event_manifest.json')

    def predict_event(row):
        nonlocal history
        event,start=row['event'],row['start_ms'];release_before(start);source=base/'events'/str(event)
        base_seal=verify_seal(source/'event_manifest.json');audit=json.loads((source/'audit.json').read_text())
        if base_seal['run_id']!=cfg['base_run_id'] or base_seal['event']!=event or base_seal['query_start_ms']!=start or base_seal['end_ms']!=row['end_ms'] or audit['current_outcome_released'] is not False:raise ValueError('Wrong cached base event clock')
        expected={r['event']:r for r in data.event_index.values() if r['end_ms']<start}
        if {r['event'] for r in audit['released_history']}!=set(expected):raise ValueError('Wrong base history cohort')
        for r in audit['released_history']:
            if any(r[k]!=expected[r['event']][k] for k in ['start_ms','end_ms','sha256']):raise ValueError('Wrong base history hash')
        raw=pd.read_csv(source/'predictions.csv');frame=raw[INPUT_COLUMNS].copy();frame['base_prediction']=raw.predicted
        wanted=case_tables[row['split']].query('event==@event').set_index('case_id')
        if set(frame.case_id)!=set(wanted.index) or frame.case_id.duplicated().any():raise ValueError('Frozen base case panel changed')
        for r in frame.itertuples():
            c=wanted.loc[r.case_id]
            if r.rank!=c['rank'] or r.cutoff!=c.cutoff or r.start_ms!=c.start_ms or r.end_ms!=c.end_ms or not np.isclose(r.current,c.current,rtol=1e-12,atol=1e-6):raise ValueError('Base prefix differs from frozen case')
        if history is None:history=frame.iloc[:0].copy();history['log_error']=pd.Series(dtype=float)
        directory=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        scores,features,details=models.predict(history.copy(deep=True),frame,query_start_ms=start,model_directory=directory/'models',ranks=cfg['ranks'])
        if set(scores)!=set(METHODS):raise ValueError('Method menu changed')
        rows=[]
        for method,values in scores.items():
            for case,value in zip(frame.to_dict('records'),values):
                rows.append({k:case[k] for k in ['case_id','event','rank','phase','current']}|{'method':method,'predicted':float(value),'log_bias':float(np.log(case['base_prediction']/value))})
        pd.DataFrame(rows).to_csv(directory/'predictions.csv',index=False);features.to_csv(directory/'features.csv',index=False);history.to_csv(directory/'released_residuals.csv',index=False)
        write_new_json(directory/'audit.json',{'event':event,'split':row['split'],'query_start_ms':start,'current_outcome_released':False,
            'released_forecasts':list(released.values()),'base_event_manifest_sha256':sha256(source/'event_manifest.json'),
            'base_history':audit['released_history'],'selection_sha256':sha256(run/'selection.json') if row['split']=='test' else None})
        seal_directory(directory,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],'query_start_ms':start,'end_ms':row['end_ms'],'cases':len(frame),'methods':METHODS})
        directory.rename(run/'events'/str(event));print(json.dumps({'sealed_event':event,'split':row['split'],'history_events':len(released),'cases':len(frame),'methods':len(scores)}),flush=True)
        return pd.DataFrame(rows)

    for split in ['train','validation','test']:
        events=[r for r in queries if r['split']==split]
        if split=='test':
            release_before(events[0]['start_ms']);truth=history.set_index('case_id');scores={}
            for method in CANDIDATES:
                p=predictions['validation'].query('method==@method');r=truth.loc[p.case_id]
                actual=np.exp(np.log(r.base_prediction.to_numpy())-r.log_error.to_numpy())
                error=pd.DataFrame({'event':p.event.to_numpy(),'ape':np.abs(p.predicted.to_numpy()/actual-1)*100})
                scores[method]=float(error.groupby('event').ape.mean().mean())
            selected=min(CANDIDATES,key=lambda k:(scores[k],CANDIDATES.index(k)))
            write_new_json(run/'selection.json',{'run_id':cfg['run_id'],'selected_recipe':selected,'candidates':CANDIDATES,'methods':METHODS,'validation_scores':scores,
                'criterion':cfg['selection'],'test_labels_used_for_selection':False,'sealed_before_test_start_ms':events[0]['start_ms'],
                'validation_event_seals':{str(r['event']):sha256(run/'events'/str(r['event'])/'event_manifest.json') for r in queries if r['split']=='validation'}})
        predictions[split]=pd.concat([predict_event(r) for r in events],ignore_index=True)
        ids=set(case_tables[split].query('event in @cfg["warm_forecast_events"]').case_id) if split=='train' else set(case_tables[split].case_id)
        for _,part in predictions[split].groupby('method'):
            if set(part.case_id)!=ids or part.case_id.duplicated().any():raise ValueError('Fixed case coverage changed')
        predictions[split].to_csv(run/('warm_predictions.csv' if split=='train' else f'{split}_predictions.csv'),index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during calibration')
    seal_directory(run,'blind_manifest.json',{'protocol':cfg['protocol'],'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'dataset_directory':cfg['dataset_directory'],'methods':METHODS,'selected_recipe':selected,'reference_method':'ewma_reference'})
    return json.loads((run/'selection.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DEFAULT);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--base-run',type=Path,default=BASE);p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(json.dumps(run(MultilineDataset(a.dataset),a.run,base_run=a.base_run,resume=a.resume),ensure_ascii=False))
