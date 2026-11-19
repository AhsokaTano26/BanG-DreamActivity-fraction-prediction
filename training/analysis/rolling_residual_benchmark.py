"""Causal V3 correction experiments over a sealed rolling-model forecast cache."""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.current_benchmark import load_dataset
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import OnlineBias, verify_seal, seal_directory, write_new_json
from analysis.rolling_dataset import RollingDataset, DEFAULT
from analysis import rolling_residual_stats as stats
from analysis import rolling_residual_models as models

BASE_RUN=ROOT/'analysis/rolling_results/v3_dense_full_retrain_20261115'
RUN=ROOT/'analysis/rolling_results/v3_residual_exploration_20261115'
DATASET_ID='85f5924e0b90deb0e1a30f3aa27d2237640cca4d968d96612dda8fee1d9c72ec'
CANDIDATES=['uncorrected',*stats.CONFIGS,*models.CONFIGS]
METHODS=CANDIDATES+[f'residual_mlp_seed{s}' for s in [42,123,2026]]
FRAME_COLUMNS=['case_id','event','rank','phase','progress','start_ms','end_ms','current','base_prediction','ewma_bias','event_type','band_group','duration_h','remaining_h','log_base_growth','rate_24h_relative','rate_6h_relative']
PRED_COLUMNS=['case_id','event','rank','phase','method','current','predicted','log_bias']


def aligned_base(frame, ids):
    if frame.case_id.duplicated().any() or set(frame.case_id)!=set(ids) or len(ids)!=len(set(ids)):
        raise ValueError('Base case identities differ from current blind prefixes')
    values=frame.set_index('case_id').loc[ids,'predicted'].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid base predictions')
    return values


def final_predictions(current, base, biases):
    current,base=np.asarray(current),np.asarray(base)
    result={}
    for method,bias in biases.items():
        bias=np.asarray(bias,float)
        if bias.shape!=base.shape or not np.isfinite(bias).all() or np.any(abs(bias)>.3+1e-12):
            raise ValueError('Invalid total log correction')
        result[method]=np.maximum(current,np.minimum(current*np.exp(4),base*np.exp(-bias)))
    seedkeys=[f'residual_mlp_seed{s}' for s in [42,123,2026]]
    if any(k in result for k in seedkeys):
        if not all(k in result for k in seedkeys):raise ValueError('All three neural seed predictions required')
        result['residual_mlp']=np.mean([result[k] for k in seedkeys],axis=0)
    return result


def _config(data,base_run,methods,allow_fixture):
    base_seal=verify_seal(base_run/'blind_manifest.json')
    base_config=json.loads((base_run/'run_manifest.json').read_text())
    if base_seal['dataset_id']!=data.manifest['dataset_id'] or base_config['run_id']!=base_seal['run_id']:
        raise ValueError('Base cache uses a different dataset/run')
    if not allow_fixture and (data.manifest['dataset_id']!=DATASET_ID or base_config['allow_fixture']):
        raise ValueError('Formal exploration requires the dense V3 formal base')
    if methods is not None and not allow_fixture:raise ValueError('Formal runs require the complete preregistered menu')
    methods=list(METHODS if methods is None else methods)
    if len(set(methods))!=len(methods) or not set(methods)<=set(METHODS) or not {'uncorrected','ewma_reference'}<=set(methods):
        raise ValueError('Invalid correction menu')
    sources=local_sources([Path(__file__)])
    cfg={'protocol':'rolling_residual_v3','dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
         'base_run':str(base_run.resolve()),'base_run_id':base_seal['run_id'],'base_blind_sha256':sha256(base_run/'blind_manifest.json'),
         'methods':methods,'candidates':[m for m in CANDIDATES if m in methods],
         'statistical_recipes':stats.CONFIGS,'learned_recipes':models.CONFIGS,
         'scored_splits':['validation','test'],'feedback_scope':'Only strictly completed earlier forecast events; no train resubstitution residuals',
         'event_clock':'One start-frozen residual training history for every prefix of an event',
         'base_cache_policy':'Reuse identical causal uncorrected rolling_context forecasts; correction-only ablation',
         'selection':'Minimum validation event-macro MAPE; fixed preregistered candidate order breaks ties',
         'ensemble':'Arithmetic mean of three bounded final scores, never mean log bias',
         'exposure':'Previously explored test and repeatedly used validation; new results are exploratory, not untouched confirmation',
         'metadata_caveat':data.manifest['protocol']['metadata_caveat'],'allow_fixture':allow_fixture,
         'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','torch','joblib','scipy']},
         'source_hashes':{name:sha256(ROOT/name) for name in sources},
         'specification_sha256':sha256(ROOT/'docs/superpowers/specs/2026-11-15-rolling-residual-design.md')}
    cfg['run_id']=identity(cfg)
    return cfg


def _frame(samples,metadata,values,state):
    rows=[]
    for sample,value in zip(samples,values):
        c=sample['curve'];progress=float(sample['progress']);phase=int(np.searchsorted([.25,.9],progress,side='right'))
        rows.append({'case_id':sample['case_id'],'event':c.event,'rank':c.rank,'phase':phase,'progress':progress,
                     'start_ms':c.start,'end_ms':c.end,'current':float(sample['score']),'base_prediction':float(value),
                     'ewma_bias':state.bias(c.rank,phase),'event_type':c.kind,'band_group':metadata.band_only(c)['band_group'],
                     'duration_h':c.duration/3600000,'remaining_h':(c.end-sample['now'])/3600000,
                     'log_base_growth':float(np.log(value/sample['score'])),
                     **{name:float(sample['features'].get(name,np.nan)) for name in ['rate_24h_relative','rate_6h_relative']}})
    return pd.DataFrame(rows,columns=FRAME_COLUMNS)


def run_experiment(data,run,*,base_run=BASE_RUN,allow_fixture=False,methods=None,resume=False):
    run,base_run=Path(run),Path(base_run)
    cfg=_config(data,base_run,methods,allow_fixture)
    if run.exists():
        if not resume:raise FileExistsError('Preserve previous run and use a new directory')
        if json.loads((run/'run_manifest.json').read_text())!=cfg:raise ValueError('Changed source/configuration cannot resume')
        if not (run/'blind_manifest.json').exists():raise FileExistsError('Incomplete causal run preserved; new run required')
        verify_seal(run/'blind_manifest.json')
        return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True);(run/'events').mkdir()
    write_new_json(run/'run_manifest.json',cfg)
    for name in cfg['source_hashes']:
        dest=run/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    for name in ['specs/2026-11-15-rolling-residual-design.md','plans/2026-11-15-rolling-residual.md']:
        shutil.copyfile(ROOT/'docs/superpowers'/name,run/Path(name).name)
    metadata=EventFeatures.load(data.metadata_directory)
    queries=sorted([r for r in data.event_index.values() if r['split']!='train'],key=lambda r:(r['start_ms'],r['event']))
    samples_by_event=defaultdict(list)
    for split in ['validation','test']:
        for sample in data.inputs(split):samples_by_event[sample['curve'].event].append(sample)
    state=OnlineBias();released={};history=pd.DataFrame(columns=FRAME_COLUMNS+['log_error'])
    predictions={}

    def release_before(start):
        nonlocal history
        for row in sorted(queries,key=lambda r:(r['end_ms'],r['event'])):
            event=row['event']
            if row['end_ms']>=start or event in released:continue
            directory=run/'events'/str(event)
            receipt=verify_seal(directory/'event_manifest.json')
            if receipt['event']!=event or receipt['run_id']!=cfg['run_id']:raise ValueError('Feedback prediction identity changed')
            curves=data.reveal_event(event,before_ms=start)
            finals={c.rank:c.final for c in curves}
            frame=pd.read_csv(directory/'features.csv')
            raw=pd.read_csv(directory/'predictions.csv');raw=raw[raw.method=='uncorrected']
            labels={r.case_id:finals[int(r.rank)] for r in raw.itertuples()}
            state.observe(raw,labels,event=event,end_ms=row['end_ms'])
            frame['log_error']=[float(np.log(r.base_prediction/finals[int(r.rank)])) for r in frame.itertuples()]
            history=pd.concat([history,frame],ignore_index=True) if len(history) else frame
            released[event]={'event':event,'start_ms':row['start_ms'],'end_ms':row['end_ms'],
                             'event_seal_sha256':sha256(directory/'event_manifest.json'),'outcome_vault_sha256':row['sha256']}

    def predict_event(row):
        event,start=row['event'],row['start_ms'];release_before(start)
        samples=samples_by_event[event];ids=[s['case_id'] for s in samples]
        cached=base_run/'events'/str(event)
        seal=verify_seal(cached/'event_manifest.json')
        audit=json.loads((cached/'audit.json').read_text())
        if seal['event']!=event or seal['query_start_ms']!=start or seal['run_id']!=cfg['base_run_id']:
            raise ValueError('Base event cache identity differs')
        if any(r['end_ms']>=start for r in audit['released_history']):raise ValueError('Base cache contains illegal history')
        old=pd.read_csv(cached/'predictions.csv')
        base=aligned_base(old[old.method=='rolling_context'],ids)
        target=_frame(samples,metadata,base,state)
        directory=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        biases={'uncorrected':np.zeros(len(target))};diagnostics={}
        for name in stats.CONFIGS:
            if name in cfg['methods']:
                biases[name],diagnostics[name]=stats.predict_bias(name,history.copy(deep=True),target.copy(deep=True),query_start_ms=start)
        for name in models.CONFIGS:
            if name in cfg['candidates']:
                values,diag=models.fit_predict(name,history.copy(deep=True),target.copy(deep=True),query_start_ms=start,model_directory=directory/name)
                biases.update(values);diagnostics[name]=diag
        values=final_predictions(target.current.to_numpy(),base,biases)
        if set(values)!=set(cfg['methods']):raise ValueError('Reported correction method coverage differs')
        old_ewma=aligned_base(old[old.method=='rolling_context_ewma'],ids)
        if not np.allclose(values['ewma_reference'],old_ewma,rtol=1e-12,atol=1e-6):raise ValueError('Reference EWMA replay does not match sealed baseline')
        records=[]
        for method,predicted in values.items():
            for c,value,raw in zip(target.to_dict('records'),predicted,base):
                records.append({k:c[k] for k in ['case_id','event','rank','phase','current']}|{'method':method,'predicted':float(value),'log_bias':float(np.log(raw/value))})
        target.to_csv(directory/'features.csv',index=False)
        pd.DataFrame(records,columns=PRED_COLUMNS).to_csv(directory/'predictions.csv',index=False)
        # Residual feedback rows are already causally released; no raw full curves.
        history.to_csv(directory/'released_residuals.csv',index=False)
        write_new_json(directory/'diagnostics.json',diagnostics)
        write_new_json(directory/'audit.json',{'event':event,'split':row['split'],'query_start_ms':start,
             'released_forecasts':list(released.values()),'feedback_events':len(released),
             'base_event_manifest_sha256':sha256(cached/'event_manifest.json'),
             'reference_ewma_max_relative_error':float(np.max(np.abs(values['ewma_reference']/old_ewma-1))),
             'selection_sha256':sha256(run/'selection.json') if row['split']=='test' else None,
             'current_outcome_released':False})
        seal_directory(directory,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],
             'query_start_ms':start,'end_ms':row['end_ms'],'cases':len(samples),'methods':cfg['methods']})
        directory.rename(run/'events'/str(event))
        print(json.dumps({'sealed_event':event,'split':row['split'],'feedback_events':len(released),'methods':len(values)}),flush=True)
        return pd.DataFrame(records,columns=PRED_COLUMNS)

    for split in ['validation','test']:
        rows=[r for r in queries if r['split']==split]
        if split=='test':
            release_before(rows[0]['start_ms'])
            val=predictions['validation'];scores={}
            for method in cfg['candidates']:
                part=val[val.method==method].copy();truth=history.set_index('case_id')
                if not set(part.case_id)<=set(truth.index):raise ValueError('Validation outcomes not completed before test start')
                actual=np.exp(np.log(truth.loc[part.case_id,'base_prediction'].to_numpy(float))-truth.loc[part.case_id,'log_error'].to_numpy(float))
                part['ape']=np.abs(part.predicted.to_numpy()/actual-1)*100
                scores[method]=float(part.groupby('event').ape.mean().mean())
            chosen=min(cfg['candidates'],key=lambda m:(scores[m],cfg['candidates'].index(m)))
            write_new_json(run/'selection.json',{'run_id':cfg['run_id'],'selected_recipe':chosen,'validation_scores':scores,
                 'criterion':cfg['selection'],'methods':cfg['methods'],'candidates':cfg['candidates'],
                 'test_labels_used_for_selection':False,'sealed_before_test_start_ms':rows[0]['start_ms'],
                 'validation_event_seals':{str(r['event']):sha256(run/'events'/str(r['event'])/'event_manifest.json') for r in queries if r['split']=='validation'}})
        predictions[split]=pd.concat([predict_event(row) for row in rows],ignore_index=True)
        expected=set(data.case_index(split).case_id)
        for method,part in predictions[split].groupby('method'):
            if part.case_id.duplicated().any() or set(part.case_id)!=expected:raise ValueError('Frozen case coverage differs')
        predictions[split].to_csv(run/f'{split}_predictions.csv',index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during experiment: '+name)
    seal_directory(run,'blind_manifest.json',{'protocol':'rolling_residual_v3','run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'dataset_directory':cfg['dataset_directory'],'methods':cfg['methods'],'scored_splits':['validation','test'],
        'reference_method':'ewma_reference','selected_recipe':chosen,'base_blind_sha256':cfg['base_blind_sha256']})
    return json.loads((run/'selection.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--base-run',type=Path,default=BASE_RUN);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(run_experiment(load_dataset('v3'),a.run,base_run=a.base_run,resume=a.resume),ensure_ascii=False))
