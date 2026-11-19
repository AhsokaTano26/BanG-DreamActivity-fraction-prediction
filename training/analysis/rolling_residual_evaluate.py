"""Independent sealed scoring and activity-paired uncertainty for V3 residuals."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.rolling_dataset import RollingDataset
from analysis.temporal_evaluate import score_temporal,aligned_frame


def intervals(cases,labels,predictions,event_order,*,draws=5000):
    errors={}
    for method,part in predictions.groupby('method'):
        frame=aligned_frame(cases,labels,part)
        frame['ape']=np.abs(frame.predicted/frame.actual-1)*100
        grouped=frame.groupby('event').ape.mean()
        if set(grouped.index)!=set(event_order) or len(set(event_order))!=len(event_order):raise ValueError('Distinct frozen activity panel required')
        errors[method]=grouped.loc[event_order].to_numpy()
    reference=errors['ewma_reference'];n=len(reference);result=[]
    for method,error in errors.items():
        delta=error-reference;record={'method':method,'reference':'ewma_reference','events':n,
            'method_minus_reference_mape_points':float(delta.mean()),'improved_events':int((delta<0).sum()),
            'event_order':event_order,'event_differences':delta.tolist()}
        if n>1:
            rng=np.random.default_rng(42);boot=delta[rng.integers(0,n,(draws,n))].mean(axis=1)
            record['paired_event_95_ci']=np.quantile(boot,[.025,.975]).tolist()
            record['circular_block_95_ci']={}
            for block in [3,5]:
                rng=np.random.default_rng(42);starts=rng.integers(0,n,(draws,int(np.ceil(n/block))))
                indices=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
                record['circular_block_95_ci'][str(block)]=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()
        result.append(record)
    return result


def evaluate(run,output=None):
    run=Path(run);seal=verify_seal(run/'blind_manifest.json')
    if seal['protocol']!='rolling_residual_v3':raise ValueError('Expected residual exploration protocol')
    data=RollingDataset(seal['dataset_directory'])
    if data.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Different data')
    output=Path(output or run);receiptpath=output/'evaluation_manifest.json'
    sources=local_sources([Path(__file__)])
    ident={'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'dataset_id':seal['dataset_id'],
           'source_hashes':{n:sha256(ROOT/n) for n in sources},'python':platform.python_version(),
           'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','catboost','joblib','httpx']},
           'draws':5000,'seed':42,'reference':'ewma_reference','block_lengths':[3,5]}
    eid=identity(ident)
    if receiptpath.exists():
        receipt=verify_seal(receiptpath)
        if receipt['evaluation_id']!=eid:raise FileExistsError('Scorer identity differs; use new --output')
        return json.loads((output/'evaluation.json').read_text())
    names=['evaluation.json','metrics.csv','by_rank.csv','by_stage.csv','by_stage_rank.csv','by_calendar.csv','by_type.csv','by_band.csv','paired_comparisons.json','evaluator_sources']
    if any((output/name).exists() for name in names):raise FileExistsError('Scoring destination contains preserved outputs')
    selection=json.loads((run/'selection.json').read_text())
    if selection['selected_recipe']!=seal['selected_recipe'] or set(selection['methods'])!=set(seal['methods']):raise ValueError('Selection differs from seal')
    result={'evaluation_id':eid,'dataset_id':seal['dataset_id'],'run_id':seal['run_id'],'selection':selection,'splits':{},
            'exposure':'Exploratory follow-up on already exposed tests and reused validation; not untouched confirmation',
            'metadata_caveat':data.manifest['protocol']['metadata_caveat'],
            'train_scope':'67 initial base-model training events are warmup, never residual resubstitution scores'}
    all_rows=[];rank_rows=[];stage_rows=[];stage_rank_rows=[];calendar_rows=[];type_rows=[];band_rows=[];comparison=[]
    for split in ['validation','test']:
        cases=data.case_index(split);members=data.memberships(split);labels=data.labels(split,allow_test=split=='test')
        predictions=pd.read_csv(run/f'{split}_predictions.csv')
        if set(predictions.method)!=set(seal['methods']):raise ValueError('Method coverage differs')
        scores={}
        for method,part in predictions.groupby('method'):
            score=score_temporal(cases,members,labels,part,stage_catalog=data.manifest['protocol']['stages']);scores[method]=score
            tags={'split':split,'method':method,'selected':method==selection['selected_recipe']}
            all_rows.append(tags|score['overall'])
            rank_rows.extend(tags|{'rank':rank}|value for rank,value in score['by_rank'].items())
            stage_rows.extend(tags|{'stage':stage}|value for stage,value in score['by_stage'].items())
            stage_rank_rows.extend(tags|value for value in score['stage_rank'])
            calendar_rows.extend(tags|value for value in score['stage_calendar'])
            type_rows.extend(tags|{'event_type':kind}|value for kind,value in score.get('by_event_type',{}).items())
            band_rows.extend(tags|{'band_group':band}|value for band,value in score.get('by_band_group',{}).items())
        result['splits'][split]=scores
        if split=='validation':
            for method,value in selection['validation_scores'].items():
                if not np.isclose(value,scores[method]['overall']['event_macro_mape_pct'],atol=1e-9,rtol=0):raise ValueError('Validation scores differ from frozen selection')
        if split=='test':
            events=sorted([r for r in data.event_index.values() if r['split']=='test'],key=lambda r:(r['start_ms'],r['event']))
            comparison=intervals(cases,labels,predictions,[r['event'] for r in events])
    result['paired_comparisons']=comparison
    output.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix='.residual-score-',dir=output.parent))
    try:
        write_new_json(staging/'evaluation.json',result)
        write_new_json(staging/'paired_comparisons.json',{'draws':5000,'seed':42,'reference':'ewma_reference','comparisons':comparison})
        for name,rows in [('metrics',all_rows),('by_rank',rank_rows),('by_stage',stage_rows),('by_stage_rank',stage_rank_rows),('by_calendar',calendar_rows),('by_type',type_rows),('by_band',band_rows)]:
            pd.DataFrame(rows).to_csv(staging/f'{name}.csv',index=False)
        for source in sources:
            path=staging/'evaluator_sources'/source;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,path)
        receipt={'evaluation_id':eid,'identity':ident,'artifacts':{str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
        receipt['seal_id']=identity(receipt);write_new_json(staging/'evaluation_manifest.json',receipt)
        output.mkdir(parents=True,exist_ok=True)
        for name in names+['evaluation_manifest.json']:
            if (output/name).exists():raise FileExistsError('Scoring output appeared')
            (staging/name).rename(output/name)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(pd.DataFrame(all_rows).query("split == 'test'")[['method','selected','event_macro_mape_pct']].to_string(index=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();evaluate(a.run,a.output)
