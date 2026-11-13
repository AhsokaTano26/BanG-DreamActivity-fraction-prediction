"""Independent scoring by forecast origin; never select/refit models on answers."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import SPLITS,sha256,write_json
from analysis.holdout_evaluate import metrics
from analysis.multirank_dataset import identity
from analysis.temporal_dataset import TemporalDataset,STAGES


def measured(frame):
    if frame.empty:return {'event_macro_mape_pct':None,'case_mape_pct':None,'p90_ape_pct':None,
        'mae_ep':None,'rmse_ep':None,'cases':0,'events':0,'median_observation_age_minutes':None}
    result=metrics(frame)
    result['median_observation_age_minutes']=float(frame.observation_age_minutes.median())
    return result


def aligned_frame(cases,labels,predictions):
    expected=set(cases.case_id)
    if cases.case_id.duplicated().any() or predictions.case_id.duplicated().any() or set(predictions.case_id)!=expected or set(labels)!=expected:
        raise ValueError('Predictions and labels must match frozen case IDs exactly')
    # Group membership comes exclusively from the frozen case index.
    frame=cases.merge(predictions[['case_id','predicted']],on='case_id',how='left',validate='one_to_one')
    frame['actual']=frame.case_id.map(labels)
    if not np.isfinite(frame[['actual','predicted']]).all().all() or (frame[['actual','predicted']]<=0).any().any():raise ValueError('Invalid score values')
    return frame


def score_temporal(cases,members,labels,predictions,*,stage_catalog=STAGES):
    frame=aligned_frame(cases,labels,predictions)
    if members[['case_id','stage_id']].duplicated().any() or set(members.case_id)!=set(cases.case_id):raise ValueError('Invalid stage membership')
    catalog={s['stage_id']:s['axis'] for s in stage_catalog}
    if any(catalog.get(r.stage_id)!=r.axis for r in members.itertuples()):raise ValueError('Stage catalog mismatch')
    expanded=members.merge(frame,on='case_id',how='left',validate='many_to_one')
    if expanded[['event','rank','stage_id']].duplicated().any():raise ValueError('Multiple cases for one event/rank/stage')
    by_stage={s:measured(expanded[expanded.stage_id==s]) for s in catalog}
    fresh={s:measured(expanded[(expanded.stage_id==s)&(expanded.observation_age_minutes<=15)]) for s in catalog}
    panels={}
    for axis in dict.fromkeys(catalog.values()):
        stages=[s for s,a in catalog.items() if a==axis];part=expanded[expanded.axis==axis]
        counts=part.groupby(['event','rank']).stage_id.nunique();pairs=set(counts[counts==len(stages)].index.tolist())
        mask=[(int(e),int(r)) in pairs for e,r in zip(part.event,part['rank'])]
        common=part.loc[mask]
        panels[axis]={'event_rank_pairs':len(pairs),'events':len({e for e,r in pairs}),
            'required_stages':stages,'by_stage':{s:measured(common[common.stage_id==s]) for s in stages}}
    result={'overall':measured(frame),'by_stage':by_stage,'fresh_15min_by_stage':fresh,'common_panels':panels,
        'by_rank':{str(k):measured(g) for k,g in frame.groupby('rank')},
        'stage_rank':[],'stage_calendar':[]}
    for (stage,rank),group in expanded.groupby(['stage_id','rank']):result['stage_rank'].append({'stage_id':stage,'rank':int(rank),**measured(group)})
    if 'calendar_quarter' in frame:
        for (stage,quarter),group in expanded.groupby(['stage_id','calendar_quarter']):
            result['stage_calendar'].append({'stage_id':stage,'calendar_quarter':quarter,**measured(group)})
    for column in ['event_type','band_group']:
        if column in frame:result['by_'+column]={str(k):measured(g) for k,g in frame.groupby(column)}
    return result


def paired_bootstrap(cases,members,labels,predictions,selected,*,draws=2000):
    rows=[];rng=np.random.default_rng(42)
    for stage in ['remaining_24h','remaining_6h','remaining_1h','remaining_0.25h']:
        ids=set(members.loc[members.stage_id==stage,'case_id']);subset=cases[cases.case_id.isin(ids)]
        if subset.empty:continue
        stage_labels={c:labels[c] for c in ids};errors={}
        for method in [selected,'hold_current','linear']:
            p=predictions[(predictions.method==method)&predictions.case_id.isin(ids)]
            frame=aligned_frame(subset,stage_labels,p)
            frame['ape']=abs(frame.predicted/frame.actual-1)*100
            errors[method]=frame.groupby('event').ape.mean().sort_index()
        for reference in ['hold_current','linear']:
            delta=(errors[selected]-errors[reference]).to_numpy();n=len(delta)
            boot=delta[rng.integers(0,n,(draws,n))].mean(axis=1) if n>=2 else None
            rows.append({'stage_id':stage,'selected':selected,'reference':reference,'events':n,
                'selected_minus_reference_mape_points':float(delta.mean()),
                'ci_low':float(np.quantile(boot,.025)) if boot is not None else None,
                'ci_high':float(np.quantile(boot,.975)) if boot is not None else None})
    return rows


def verify_predictions(run):
    run=Path(run);seal=json.loads((run/'blind_manifest.json').read_text())
    for name,digest in seal['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('Sealed prediction artifact changed: '+name)
    return seal


def evaluate(run,output=None):
    run=Path(run);seal=verify_predictions(run);data=TemporalDataset(seal['dataset_directory'])
    if data.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Dataset differs from frozen predictions')
    output=Path(output or run);receipt=output/'evaluation_manifest.json'
    sources=['analysis/temporal_evaluate.py','analysis/temporal_dataset.py','analysis/holdout_evaluate.py',
             'analysis/holdout_dataset.py','analysis/multirank_dataset.py','analysis/backtest.py']
    ident={'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'dataset_id':data.manifest['dataset_id'],
        'sources':{n:sha256(ROOT/n) for n in sources},'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas']},'bootstrap_seed':42,'bootstrap_draws':2000}
    evaluation_id=identity(ident)
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['evaluation_id']!=evaluation_id:raise FileExistsError('Different evaluator; choose new --output')
        for name,digest in saved['artifacts'].items():
            if sha256(output/name)!=digest:raise ValueError('Scored artifact changed')
        return json.loads((output/'evaluation.json').read_text())
    names=['evaluation.json','stage_metrics.csv','stage_rank_metrics.csv','stage_calendar_metrics.csv','paired_bootstrap.csv','evaluator_sources']
    if any((output/n).exists() for n in names):raise FileExistsError('Scoring destination already contains outputs')
    result={'evaluation_id':evaluation_id,'dataset_id':data.manifest['dataset_id'],'splits':{},
        'exposure':data.manifest['protocol']['exposure'],'metadata_caveat':data.manifest['protocol']['metadata_caveat']}
    stage_rows=[];rank_rows=[];calendar_rows=[];bootstrap=[]
    for split in SPLITS:
        cases=data.case_index(split);members=data.memberships(split)
        labels=data.labels(split,allow_test=split=='test')
        predictions=pd.read_csv(run/f'{split}_predictions.csv')
        if set(predictions.method)!=set(seal['methods']):raise ValueError('Method set differs from sealed configuration')
        scores={}
        for method,group in predictions.groupby('method'):
            score=score_temporal(cases,members,labels,group);scores[method]=score
            for stage,metric in score['by_stage'].items():stage_rows.append({'split':split,'method':method,'stage_id':stage,**metric})
            rank_rows.extend({'split':split,'method':method,**row} for row in score['stage_rank'])
            calendar_rows.extend({'split':split,'method':method,**row} for row in score['stage_calendar'])
        result['splits'][split]=scores
        if split=='test':bootstrap=paired_bootstrap(cases,members,labels,predictions,seal['reference_method'])
    result['paired_bootstrap']=bootstrap
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'evaluation.json',result)
    for name,rows in [('stage_metrics.csv',stage_rows),('stage_rank_metrics.csv',rank_rows),('stage_calendar_metrics.csv',calendar_rows),('paired_bootstrap.csv',bootstrap)]:
        pd.DataFrame(rows).to_csv(output/name,index=False)
    (output/'evaluator_sources').mkdir()
    for name in sources:shutil.copyfile(ROOT/name,output/'evaluator_sources'/Path(name).name)
    files=[n for n in names if n!='evaluator_sources']+[str(p.relative_to(output)) for p in (output/'evaluator_sources').iterdir()]
    write_json(receipt,{'evaluation_id':evaluation_id,'identity':ident,'artifacts':{n:sha256(output/n) for n in files}})
    print(json.dumps({'evaluation_id':evaluation_id,'test':{m:s['overall'] for m,s in result['splits']['test'].items()}},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();evaluate(a.run,a.output)
