"""Trusted train-only coordinator for the frozen B sharing experiment.

The outcome vault is coordinator-owned. Only released, already-issued expert
records cross into adaptive_sharing; current model frames are strict allowlists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import adaptive_sharing as model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def verify_inputs(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or sha256(root/name)!=expected:
            raise ValueError('Input SHA mismatch: '+str(name))
    return manifest


def load_train_vault(path):
    return pd.read_csv(path,dtype={'event_type':str,'band_group':str},low_memory=False)


def metrics(frame):
    if frame.empty:raise ValueError('Cannot score an empty panel')
    scored=frame.copy();difference=scored.prediction-scored.actual
    scored['ape']=100*np.abs(difference/scored.actual)
    scored['log_squared']=np.log(scored.prediction/scored.actual)**2
    return {'events':int(scored.event.nunique()),'cases':len(scored),
        'event_macro_mape':float(scored.groupby('event').ape.mean().mean()),
        'rank_equal_event_macro_mape':float(scored.groupby(['event','rank']).ape.mean().groupby('event').mean().mean()),
        'case_mape':float(scored.ape.mean()),'mae':float(np.abs(difference).mean()),
        'rmse':float(np.sqrt(np.mean(difference**2))),
        'p90_ape':float(scored.ape.quantile(.9)),
        'event_macro_log_squared':float(scored.groupby('event').log_squared.mean().mean())}


def _block(index,schedule):
    return ('warmup' if index<schedule['warmup'] else 'selection' if
            index<schedule['warmup']+schedule['selection'] else 'diagnostic')


def _source_fingerprints():
    root=Path(__file__).resolve().parent
    return {p.name:sha256(p) for p in sorted(root.iterdir()) if p.suffix in ('.py','.md','.json')}


def run_replay(outcomes,queries,configuration,output,*,source_metadata=None,allow_fixture=False):
    """Issue all predictions before releasing each event's outcomes to models."""
    started=time.monotonic();output=Path(output);cfg=dict(configuration)
    schedule=cfg['schedule'];candidates=cfg['candidates'];names=[item['name'] for item in candidates]
    if len(set(names))!=len(names) or len(names)>12:raise ValueError('Invalid frozen candidate menu')
    queries=sorted(queries,key=lambda q:(int(q['start_ms']),int(q['event'])))
    n=sum(schedule[key] for key in ('warmup','selection','diagnostic'))
    if len(queries)!=n or (not allow_fixture and (n!=35 or len(names)!=12)):
        raise ValueError('Wrong frozen query/candidate count')
    if (outcomes.case_id.duplicated().any() or outcomes.case_id.isna().any()
            or not outcomes['split'].eq('train').all()
            or set(map(int,outcomes.event))!={int(q['event']) for q in queries}):
        raise ValueError('Input must contain all unique training cases')
    if any(q.get('split')!='train' or int(q['start_ms'])<model.MIN_START_MS for q in queries):
        raise ValueError('Only dense-period training queries are permitted')
    if len({q['event'] for q in queries})!=n:raise ValueError('Repeated query event')
    output.mkdir(parents=True,exist_ok=False)
    (output/'issued').mkdir();(output/'experts').mkdir();(output/'seals').mkdir()
    sources=_source_fingerprints();metadata=dict(source_metadata or {})
    write_json(output/'configuration.json',cfg)
    write_json(output/'input_metadata.json',metadata)
    blind_columns=[column for column in model.BLIND_COLUMNS if column in outcomes]
    history=pd.DataFrame(columns=blind_columns+['log_error']+list(model.ISSUED))
    issued={};released=set();scores=[];selected=None;calls=0

    def release(clock):
        nonlocal history
        for event,entry in issued.items():
            if event in released or entry['end_ms']>=clock:continue
            raw=outcomes.loc[outcomes.event==event].set_index('case_id').reindex(entry['blind'].case_id)
            z=raw.log_error.to_numpy(float);actual=raw.actual.to_numpy(float)
            if (not np.isfinite(z).all() or not np.isfinite(actual).all() or np.any(actual<=0)
                    or not np.allclose(actual,raw.base_prediction.to_numpy(float)*np.exp(-z),rtol=1e-10,atol=1e-6)):
                raise ValueError('Invalid released outcome or log-error direction')
            # Released model history carries z and the exact expert values in the
            # immutable issuance artifact; no old reference is treated as a label.
            part=entry['blind'].copy(deep=True);part['log_error']=z
            expert=pd.read_csv(entry['expert_path'])
            if list(expert.case_id)!=list(part.case_id):raise ValueError('Issued expert order changed')
            for new,old in [('issued_ind','expert_ind'),('issued_pair','expert_pair'),
                            ('issued_adjacent','expert_adjacent'),('issued_samepanel','expert_samepanel')]:
                part[new]=expert[old].to_numpy(float)
            history=part.copy() if history.empty else pd.concat([history,part],ignore_index=True)
            pred=pd.read_csv(entry['prediction_path'])
            if sha256(entry['prediction_path'])!=entry['prediction_sha'] or sha256(entry['expert_path'])!=entry['expert_sha']:
                raise ValueError('Issued prediction/expert SHA changed before feedback')
            actual_map=pd.Series(actual,index=part.case_id)
            pred['actual']=pred.case_id.map(actual_map);scores.append(pred)
            for external,column in [('external_original_independent','expert__independent'),('uncorrected_base','base_prediction')]:
                if column not in raw:continue
                control=entry['blind'][['case_id','event','rank','phase']].copy()
                control['candidate']=external;control['block']=entry['block']
                control['prediction']=raw[column].to_numpy(float);control['actual']=actual
                scores.append(control)
            released.add(event)

    for index,query in enumerate(queries):
        if time.monotonic()-started>cfg['budgets']['wall_seconds']:
            raise TimeoutError('Frozen B wall-clock budget exceeded; partial directory preserved')
        clock=int(query['start_ms']);event=int(query['event']);release(clock)
        if index==schedule['warmup']+schedule['selection']:
            selection_events=[int(q['event']) for q in queries[schedule['warmup']:index]]
            if not set(selection_events)<=released:
                raise ValueError('All selection events must be released before diagnostic freeze')
            selection_scores=pd.concat(scores,ignore_index=True)
            selection_scores=selection_scores[selection_scores.event.isin(selection_events)&selection_scores['rank'].isin(model.PAIR)]
            values=[]
            for name in names:
                value=metrics(selection_scores[selection_scores.candidate==name])
                values.append({'candidate':name,**value})
            selected=min(range(len(values)),key=lambda i:(values[i]['event_macro_mape'],i))
            selected=names[selected]
            write_json(output/'selection_seal.json',{
                'selected_candidate':selected,'selection_events':selection_events,
                'selection_metric':'two-rank event-macro MAPE: unique-case mean within event, then equal events',
                'scores':values,'tie_break':'frozen candidate order',
                'freeze_start_ms':clock,'issued_events_before_seal':len(issued),
                'configuration_sha256':sha256(output/'configuration.json'),
                'latest_feedback_end_ms':int(history.end_ms.max()) if len(history) else None,
                'source_fingerprints':sources,'input_metadata':metadata})
        # Outcome/old-expert columns are excluded before any model call.
        blind=outcomes.loc[outcomes.event==event,blind_columns].copy(deep=True).reset_index(drop=True)
        if (not blind.start_ms.eq(clock).all() or not blind.end_ms.eq(int(query['end_ms'])).all()
                or (not allow_fixture and len(blind)!=query['target_cases'])):
            raise ValueError('Frozen query metadata mismatch')
        experts=model.issue_experts(history,blind)
        filename=f'{index+1:02d}_event{event}.csv.gz'
        expert_path=output/'experts'/filename;experts.to_csv(expert_path,index=False,float_format='%.17g')
        rows=[]
        for candidate in candidates:
            predictions=model.mix_experts(history,blind,candidate,experts);calls+=1
            predictions['candidate']=candidate['name'];predictions['block']=_block(index,schedule)
            predictions['query_index']=index+1
            predictions['selected_for_diagnostic']=bool(_block(index,schedule)=='diagnostic' and candidate['name']==selected)
            predictions=predictions.merge(blind[['case_id','start_ms','end_ms','current','base_prediction']],on='case_id',validate='one_to_one',sort=False)
            rows.append(predictions)
        prediction_path=output/'issued'/filename
        pd.concat(rows,ignore_index=True).to_csv(prediction_path,index=False,float_format='%.17g')
        entry={'blind':blind,'end_ms':int(query['end_ms']),'block':_block(index,schedule),
               'prediction_path':prediction_path,'expert_path':expert_path,
               'prediction_sha':sha256(prediction_path),'expert_sha':sha256(expert_path)}
        issued[event]=entry
        write_json(output/'seals'/f'{index+1:02d}_event{event}.json',{
            'event':event,'query_start_ms':clock,'issued_before_feedback':True,
            'prediction_sha256':entry['prediction_sha'],'experts_sha256':entry['expert_sha'],
            'history_events':sorted(map(int,history.event.unique())),
            'history_cases':len(history),'latest_history_end_ms':int(history.end_ms.max()) if len(history) else None,
            'selected_candidate':selected})
    # Terminal scoring is after every current forecast has been sealed. No model
    # is called at this terminal synthetic clock.
    release(max(int(q['end_ms']) for q in queries)+1)
    scored=pd.concat(scores,ignore_index=True)
    scored['ape']=100*np.abs(scored.prediction/scored.actual-1.)
    scored['log_squared']=np.log(scored.prediction/scored.actual)**2
    scored.to_csv(output/'scored_cases.csv.gz',index=False,float_format='%.17g')
    event_scores=scored.groupby(['block','candidate','event'],as_index=False).agg(
        cases=('case_id','size'),mape=('ape','mean'),log_squared=('log_squared','mean'))
    event_scores.to_csv(output/'event_scores.csv',index=False,float_format='%.17g')
    summaries=[]
    for block in ['warmup','selection','diagnostic','all']:
        block_frame=scored if block=='all' else scored[scored.block==block]
        for name in names+['external_original_independent','uncorrected_base']:
            one=block_frame[block_frame.candidate==name]
            if one.empty:continue
            for panel,ranks in [('all19',model.RANKS),('main_pair',model.PAIR)]:
                part=one[one['rank'].isin(ranks)]
                if part.empty:continue
                summaries.append({'block':block,'candidate':name,'panel':panel,
                                  'selected':name==selected,**metrics(part)})
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False,float_format='%.17g')
    if _source_fingerprints()!=sources:raise ValueError('B source changed during sealed run')
    files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'status':'complete','fixture':bool(allow_fixture),'query_events':n,
        'predict_calls':calls,'fixed_expert_calls':n,'selected_candidate':selected,
        'elapsed_seconds':time.monotonic()-started,'source_fingerprints':sources,
        'input_metadata':metadata,'versions':{'python':platform.python_version(),
        'numpy':np.__version__,'pandas':pd.__version__},
        'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        'files':files,'limitations':cfg.get('limitations',[])}
    write_json(output/'run_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();input_manifest=verify_inputs(args.inputs)
    config=json.loads(args.config.read_text())
    for name,expected in config['input_files'].items():
        if sha256(args.inputs/name)!=expected:raise ValueError('Preregistered input SHA mismatch')
    queries=json.loads((args.inputs/'train_queries.json').read_text())
    outcomes=load_train_vault(args.inputs/'train_residuals.csv.gz')
    metadata={'manifest':input_manifest,'manifest_sha256':sha256(args.inputs/'manifest.json'),
              'config_path':str(args.config),'config_sha256':sha256(args.config)}
    manifest=run_replay(outcomes,queries,config,args.output,source_metadata=metadata)
    print(json.dumps({k:manifest[k] for k in ['status','predict_calls','selected_candidate','elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':main()
