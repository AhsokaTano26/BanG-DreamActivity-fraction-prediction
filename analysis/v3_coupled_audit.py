"""Audit completed causal artifacts and replay saved neural weights, without test labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.rolling_dataset import RollingDataset
from analysis.temporal_evaluate import score_temporal
from analysis.v3_coupled_runner import RUN
from analysis.v3_coupled_models import replay_neural,LAGS


def ewma(values,alpha):
    result=0.
    for i,value in enumerate(values):result=float(value) if i==0 else (1-alpha)*result+alpha*float(value)
    return result


def audit(run,records):
    run,records=Path(run),Path(records)
    if (run/'evaluation_manifest.json').exists():raise ValueError('Pre-scoring audit must precede independent scoring')
    cfg=json.loads((run/'run_manifest.json').read_text());blind=verify_seal(run/'blind_manifest.json')
    data=RollingDataset(cfg['dataset_directory']);source=Path(cfg['base_run'])
    source_seal=verify_seal(source/'blind_manifest.json')
    assert source_seal['run_id']==cfg['base_run_id'] and sha256(source/'blind_manifest.json')==cfg['base_blind_sha256']
    assert sha256(ROOT/'docs/superpowers/specs/2026-11-17-coupled-adaptation-design.md')==cfg['specification_sha256']
    for name,digest in cfg['source_hashes'].items():assert sha256(ROOT/name)==sha256(run/'sources'/name)==digest,name
    protected=json.loads((records/'preservation_before.json').read_text())
    for name,digest in protected.items():assert sha256(ROOT/name)==digest,name
    queries={r['event']:r for r in data.event_index.values() if r['event'] in cfg['warm_forecast_events'] or r['split']!='train'}
    events=[];replays=0
    selection_sha=sha256(run/'selection.json')
    for event,row in sorted(queries.items(),key=lambda item:(item[1]['start_ms'],item[0])):
        directory=run/'events'/str(event);seal=verify_seal(directory/'event_manifest.json')
        a=json.loads((directory/'audit.json').read_text());history=pd.read_csv(directory/'released_residuals.csv')
        features=pd.read_csv(directory/'features.csv');pred=pd.read_csv(directory/'predictions.csv')
        assert seal['run_id']==cfg['run_id'] and seal['query_start_ms']==row['start_ms'] and seal['end_ms']==row['end_ms']
        assert a['current_outcome_released'] is False
        expected={e for e,r in queries.items() if r['end_ms']<row['start_ms']}
        assert set(history.event)==expected=={r['event'] for r in a['released_forecasts']}
        assert not len(history) or (history.end_ms<row['start_ms']).all()
        for released in a['released_forecasts']:
            assert all(released[k]==data.event_index[released['event']][k] for k in ['start_ms','end_ms','sha256'])
            assert released['prediction_seal_sha256']==sha256(run/'events'/str(released['event'])/'event_manifest.json')
        raw_expected={r['event'] for r in data.event_index.values() if r['end_ms']<row['start_ms']}
        assert {r['event'] for r in a['source_base_history']}==raw_expected
        for old in a['source_base_history']:assert all(old[k]==data.event_index[old['event']][k] for k in ['start_ms','end_ms','sha256'])
        assert a['source_event_manifest_sha256']==sha256(source/'events'/str(event)/'event_manifest.json')
        if row['split']=='test':assert a['selection_sha256']==selection_sha
        ids=set(data.case_index(row['split']).query('event==@event').case_id)
        assert set(features.case_id)==ids and not features.case_id.duplicated().any()
        assert set(pred.method)==set(cfg['methods']) and len(pred)==len(ids)*17
        for _,panel in pred.groupby('method'):assert set(panel.case_id)==ids and not panel.case_id.duplicated().any()
        assert (pred.predicted>=pred.current-1e-6).all() and (pred.predicted<=pred.current*np.exp(4)+1e-6).all()
        table=pred.pivot(index='case_id',columns='method',values='predicted')
        assert np.abs(np.log(table.div(table.uncorrected,axis=0))).to_numpy().max()<=.3+1e-10
        np.testing.assert_allclose(table.context_gate,np.mean([table[f'context_gate_seed{s}'] for s in [42,123,2026]],axis=0),rtol=1e-12)
        cached=pd.read_csv(source/'events'/str(event)/'predictions.csv')
        for ours,theirs in [('uncorrected','uncorrected'),('ewma_reference','ewma_reference'),('ewma_warm','ewma_warm'),('coupled_gap','coupled_gap'),('kernel_reference','mape_kernel')]:
            actual=pred.query('method==@ours').set_index('case_id').predicted.sort_index()
            reference=cached.query('method==@theirs').set_index('case_id').predicted.sort_index()
            np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-6)
        pairs=pd.read_csv(directory/'peer_audit.csv').query('peer_available==1')
        assert (pairs.peer_observed_at<=pairs.observed_at).all()
        for name in ['gap_covariate5','gap_covariate50','context_gate']:
            w=pd.read_csv(directory/'models'/name/'training_weights.csv')
            expected_cases=set(history.case_id if name=='context_gate' else history.loc[history['rank']==1000,'case_id'])
            assert set(w.case_id)==expected_cases
            if len(w):np.testing.assert_allclose(w.groupby('event').weight.sum(),1.,rtol=1e-12)
        # Independently rebuild the causal per-phase input states from released rows.
        means=history.groupby(['event','phase','rank'],as_index=False).agg(error=('log_error','mean'),end=('end_ms','first'),start=('start_ms','first'))
        for phase,current in features.groupby('phase'):
            p=means[means.phase==phase].sort_values(['end','start','event'])
            anchor=p[p['rank']==2000].error.to_numpy();na=len(anchor)
            pair=p.pivot(index='event',columns='rank',values='error').reindex(columns=[1000,2000]).dropna()
            pair=pair.join(p.drop_duplicates('event').set_index('event')[['end','start']]).sort_values(['end','start']).reset_index().sort_values(['end','start','event'])
            gap=(pair[1000]-pair[2000]).to_numpy();n=len(gap)
            values=[ewma(anchor,.2)*na/(na+10),ewma(gap,.6)*n/(n+5),ewma(gap,.2)*n/(n+5),
                    gap[-1] if n else 0.,np.std(gap[-5:]) if n else 0.,n,(row['start_ms']-pair.end.iloc[-1])/86400000 if n else 0.]
            for name,value in zip(LAGS,values):np.testing.assert_allclose(current[name],value,rtol=1e-12,atol=1e-12)
        # Each historic feature row must be the exact row sealed at its own start.
        for historic_event,part in history.groupby('event'):
            old=pd.read_csv(run/'events'/str(int(historic_event))/'features.csv')
            columns=list(old.columns)
            actual=part[columns].sort_values('case_id').reset_index(drop=True).copy()
            old=old[columns].sort_values('case_id').reset_index(drop=True).copy()
            # CSV infers numeric-looking single-event band labels as integers;
            # the model's category function treats 1 and "1" identically.
            for column in ['event_type','band_group']:
                actual[column]=actual[column].map(str);old[column]=old[column].map(str)
            pd.testing.assert_frame_equal(actual,old,check_dtype=False,rtol=1e-12,atol=1e-12)
        replay=replay_neural(directory/'models/context_gate',features)
        for method,values in replay.items():
            expected=pred.query('method==@method').set_index('case_id').loc[features.case_id].predicted
            np.testing.assert_allclose(values,expected,rtol=1e-12,atol=1e-6)
        replays+=1;events.append(event)
    for split in ['validation','test']:
        p=pd.read_csv(run/f'{split}_predictions.csv');ids=set(data.case_index(split).case_id)
        for _,part in p.groupby('method'):assert set(part.case_id)==ids and not part.case_id.duplicated().any()
    selection=json.loads((run/'selection.json').read_text())
    assert selection['selected_recipe']==min(cfg['candidates'],key=lambda k:(selection['validation_scores'][k],cfg['candidates'].index(k)))
    labels=data.labels('validation');p=pd.read_csv(run/'validation_predictions.csv')
    for method,value in selection['validation_scores'].items():
        score=score_temporal(data.case_index('validation'),data.memberships('validation'),labels,p.query('method==@method'))
        assert abs(score['overall']['event_macro_mape_pct']-value)<1e-9
    receipt={'status':'pass','dataset_id':data.manifest['dataset_id'],'run_id':cfg['run_id'],'audited_events':len(events),
        'neural_replay_events':replays,'preserved_old_files':len(protected),'real_test_labels_read':False,'selected_method':selection['selected_recipe'],
        'validation_mape_pct':selection['validation_scores'][selection['selected_recipe']],
        'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'selection_sha256':selection_sha,'audit_source_sha256':sha256(Path(__file__)),
        'checks':['strict histories and release seals','exact frozen panels','identical5controls','all fitted weights event-equal',
                  'historical lag/expert rows immutable','independent lag reconstruction','NN checkpoint replay and score mean','validation-only selection','production/old-file preservation']}
    write_new_json(records/'pre_scoring_audit.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=ROOT/'analysis/v3_coupled_records/20261117')
    a=p.parse_args();audit(a.run,a.records)
