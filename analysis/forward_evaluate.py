"""Final-only external scoring, preserving missing outcomes and forecast provenance."""
import argparse,importlib.metadata,json,platform,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256,write_json
from analysis.multirank_dataset import identity
from analysis.temporal_evaluate import verify_predictions,score_temporal,aligned_frame,measured
from analysis.forward_validation import ForwardDataset


def evaluate(run,output=None):
    run=Path(run);seal=verify_predictions(run);data=ForwardDataset(seal['dataset_directory'])
    if data.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Dataset changed')
    sources=['analysis/forward_evaluate.py','analysis/forward_validation.py','analysis/temporal_evaluate.py',
             'analysis/holdout_evaluate.py','analysis/holdout_dataset.py','analysis/multirank_dataset.py','analysis/backtest.py']
    ident={'prediction_seal_sha256':sha256(run/'blind_manifest.json'),'dataset_id':data.manifest['dataset_id'],
        'sources':{n:sha256(ROOT/n) for n in sources},'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas']}}
    output=Path(output or run)
    evaluation_id=identity(ident);receipt=output/'evaluation_manifest.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['evaluation_id']!=evaluation_id:raise FileExistsError('Changed evaluator; choose a separate audit run')
        for n,d in old['artifacts'].items():
            if sha256(output/n)!=d:raise ValueError('Scored output changed')
        return json.loads((output/'evaluation.json').read_text())
    if (output/'evaluation.json').exists() or (output/'evaluator_sources').exists():raise FileExistsError('Scoring output exists')
    outcomes=data.outcomes(allow=True)
    known={(int(r.event),int(r.rank)):float(r.actual) for r in outcomes.itertuples() if r.status=='available_near_end'}
    all_events=set(data.manifest['events']);scored_events={e for e in all_events if all((e,r) in known for r in [1000,2000])}
    result={'evaluation_id':evaluation_id,'mode':data.manifest['mode'],'requested_events':sorted(all_events),
        'scored_events':sorted(scored_events),'pending_events':sorted(all_events-scored_events),'full_cohort_event_mape_pct':None,
        'outcomes':outcomes.replace({np.nan:None}).to_dict('records'),'grid':{},'legacy':{},'legacy_actual_forward_forecast':[],
        'caveats':['Outside-cohort retrospective replay, not a prospective forecast of the current model.',
                   f'{len(scored_events)} completed new activities are scoreable; this small cohort cannot establish stable improvement.',
                   'Previous test cohort explored; metadata retrospective without historical-time archive.']}
    for group in ['grid','legacy']:
        file=run/f'{group}_predictions.csv'
        if not file.exists():continue
        cases=data.cases(group);predictions=pd.read_csv(file)
        if set(predictions.method)!=set(seal['methods']):raise ValueError('Method set changed')
        expected=set(cases.case_id)
        for name,p in predictions.groupby('method'):
            if p.case_id.duplicated().any() or set(p.case_id)!=expected:raise ValueError('Candidate cases differ')
            if not np.isfinite(p.predicted).all() or (p.predicted<=0).any():raise ValueError('Invalid forecast values, including pending cases')
        cases=cases[cases.event.isin(scored_events)]
        labels={r.case_id:known[int(r.event),int(r.rank)] for r in cases.itertuples()}
        for method,p in predictions.groupby('method'):
            p=p[p.case_id.isin(cases.case_id)]
            if cases.empty:result[group][method]={'overall':{'event_macro_mape_pct':None,'events':0,'cases':0}};continue
            if group=='grid':
                members=data.memberships();members=members[members.case_id.isin(cases.case_id)]
                result[group][method]=score_temporal(cases,members,labels,p,stage_catalog=data.manifest['stages'])
            else:
                frame=aligned_frame(cases,labels,p);frame['ape_pct']=abs(frame.predicted/frame.actual-1)*100
                result[group][method]={'overall':measured(frame),'rows':frame.to_dict('records')}
    if not result['pending_events']:
        result['full_cohort_event_mape_pct']=result['grid']['selected_combo']['overall']['event_macro_mape_pct']
    legacy=data.root/'legacy_forecast.json'
    if legacy.exists():
        old=json.loads(legacy.read_text());event=old['event_id']
        for row in old['predictions']:
            key=event,row['rank']
            if key not in known:continue
            actual=known[key];prediction=row['predicted_final_ep']
            result['legacy_actual_forward_forecast'].append({'event':event,'rank':row['rank'],'forecast_as_of':old['as_of'],
                'predicted':prediction,'actual':actual,'ape_pct':abs(prediction/actual-1)*100,'model':old['model']})
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'evaluation.json',result)
    (output/'evaluator_sources').mkdir()
    for source in sources:shutil.copyfile(ROOT/source,output/'evaluator_sources'/Path(source).name)
    files=[output/'evaluation.json']+list((output/'evaluator_sources').iterdir())
    write_json(receipt,{'evaluation_id':evaluation_id,'identity':ident,'artifacts':{str(p.relative_to(output)):sha256(p) for p in files}})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();result=evaluate(args.run,args.output)
    print(json.dumps({'scored_events':result['scored_events'],'pending_events':result['pending_events'],
        'grid':{k:v['overall'] for k,v in result['grid'].items()},'legacy_forward':result['legacy_actual_forward_forecast']},ensure_ascii=False,indent=2))
