"""Pre-scoring identity/clock/weight audit and saved raw/neural replays."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.multiline_dataset import MultilineDataset
from analysis.multiline_runner import RUN
from analysis.multiline_models import replay_neural
from analysis.temporal_evaluate import score_temporal

RECORDS=ROOT/'analysis/multiline_records/20261117'

def read(path):return pd.read_csv(path,float_precision='round_trip')

def audit(run=RUN,records=RECORDS):
    run,records=Path(run),Path(records)
    if (run/'evaluation_manifest.json').exists():raise ValueError('Audit before independent scoring only')
    seal=verify_seal(run/'blind_manifest.json');cfg=json.loads((run/'run_manifest.json').read_text())
    data=MultilineDataset(cfg['dataset_directory']);base=Path(cfg['base_run']);raw_cfg=json.loads((base/'run_manifest.json').read_text())
    raw_seal=verify_seal(base/'blind_manifest.json');context=Path(cfg['context_run'])
    assert raw_seal['run_id']==cfg['base_run_id'] and sha256(base/'blind_manifest.json')==cfg['base_blind_sha256']
    for root,config in [(run,cfg),(base,raw_cfg)]:
        for name,digest in config['source_hashes'].items():assert sha256(ROOT/name)==sha256(root/'sources'/name)==digest,name
    assert sha256(ROOT/'docs/superpowers/specs/2026-11-17-multiline-v3-design.md')==cfg['specification_sha256']
    protected=json.loads((records/'preservation_before.json').read_text())
    for name,digest in protected.items():assert sha256(ROOT/name)==digest,name
    x=joblib.load(base/'blind_features.joblib');all_cases=read(base/'blind_cases.csv')
    assert not all_cases.case_id.duplicated().any() and len(x)==len(all_cases)
    events={r['event']:r for r in data.event_index.values() if r['event'] in cfg['warm_forecast_events'] or r['split']!='train'}
    selected=json.loads((run/'selection.json').read_text());selection_sha=sha256(run/'selection.json')
    replayed=0;raw_replayed=0;support=[]
    for event,row in sorted(events.items(),key=lambda pair:(pair[1]['start_ms'],pair[0])):
        directory=run/'events'/str(event);event_seal=verify_seal(directory/'event_manifest.json')
        audit=json.loads((directory/'audit.json').read_text());rawdir=base/'events'/str(event);verify_seal(rawdir/'event_manifest.json')
        raw_audit=json.loads((rawdir/'audit.json').read_text());features=read(directory/'features.csv');pred=read(directory/'predictions.csv')
        history=read(directory/'released_residuals.csv');base_predictions=read(rawdir/'predictions.csv')
        assert event_seal['run_id']==cfg['run_id'] and event_seal['query_start_ms']==row['start_ms'] and event_seal['end_ms']==row['end_ms']
        assert audit['current_outcome_released'] is False and raw_audit['current_outcome_released'] is False
        expected_past={r['event']:r for r in data.event_index.values() if r['end_ms']<row['start_ms']}
        assert {r['event'] for r in raw_audit['released_history']}==set(expected_past)
        for r in raw_audit['released_history']:assert all(r[k]==expected_past[r['event']][k] for k in ['start_ms','end_ms','sha256'])
        expected_forecasts=set(expected_past)&set(events)
        assert set(history.event)==expected_forecasts=={r['event'] for r in audit['released_forecasts']}
        assert history.empty or (history.end_ms<row['start_ms']).all()
        for r in audit['released_forecasts']:
            assert all(r[k]==data.event_index[r['event']][k] for k in ['start_ms','end_ms','sha256'])
            assert r['prediction_seal_sha256']==sha256(run/'events'/str(r['event'])/'event_manifest.json')
        assert audit['base_event_manifest_sha256']==sha256(rawdir/'event_manifest.json')
        if row['split']=='test':assert audit['selection_sha256']==selection_sha
        cases=data.case_index(row['split']).query('event==@event');ids=set(cases.case_id)
        assert set(features.case_id)==ids==set(base_predictions.case_id) and not features.case_id.duplicated().any()
        assert set(pred.method)==set(cfg['methods']) and len(pred)==len(ids)*15
        for _,p in pred.groupby('method'):assert set(p.case_id)==ids and not p.case_id.duplicated().any()
        assert (pred.predicted>=pred.current-1e-6).all() and (pred.predicted<=pred.current*np.exp(4)+1e-6).all()
        table=pred.pivot(index='case_id',columns='method',values='predicted')
        assert np.abs(np.log(table.div(table.uncorrected,axis=0))).to_numpy().max()<=.3+1e-10
        np.testing.assert_allclose(table.neural_gate,np.mean([table[f'neural_gate_seed{s}'] for s in [42,123,2026]],axis=0),rtol=1e-12)
        # The issued Context panel and legacy correction remain the old controls.
        old=read(context/'events'/str(event)/'predictions.csv')
        for ours,theirs in [('uncorrected','uncorrected'),('ewma_reference','ewma_warm'),('ewma_cold','ewma_reference'),('legacy_pair','coupled_gap')]:
            a=pred.query('method==@ours and rank in [1000,2000]').set_index('case_id').predicted.sort_index()
            b=old.query('method==@theirs').set_index('case_id').predicted.sort_index()
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-6)
        # Replay each shared checkpoint without refitting or opening answer files.
        mask=all_cases.event==event;target=all_cases[mask];bundle=joblib.load(rawdir/'shared.joblib')
        assert bundle['model'].tree_count_==raw_cfg['base_recipe']['iterations']
        values=target.current.to_numpy()*np.exp(np.clip(bundle['model'].predict(bundle['preprocessor'].transform(x.loc[mask]),thread_count=1)-np.log(target.progress.to_numpy()),0,4))
        saved=base_predictions.set_index('case_id').loc[target.case_id]
        np.testing.assert_allclose(values,saved.shared_prediction,rtol=1e-12,atol=1e-6)
        shared=saved['rank'].isin([1000,2000])==False
        np.testing.assert_allclose(saved.loc[shared,'predicted'],saved.loc[shared,'shared_prediction'],rtol=1e-12)
        raw_replayed+=1
        raw_weights=read(rawdir/'training_weights.csv');expected_ids=set(all_cases[all_cases.event.isin(expected_past)].case_id)
        assert set(raw_weights.case_id)==expected_ids and not raw_weights.case_id.duplicated().any()
        mass=raw_weights.groupby('event').weight.sum();np.testing.assert_allclose(mass,mass.iloc[0],rtol=1e-12)
        neural_weights=read(directory/'models/neural_gate/training_weights.csv')
        assert set(neural_weights.case_id)==set(history.case_id)
        if len(neural_weights):np.testing.assert_allclose(neural_weights.groupby('event').weight.sum(),1.,rtol=1e-12)
        for (rank,phase),part in features.groupby(['rank','phase']):
            n=history[(history['rank']==rank)&(history.phase==phase)].event.nunique()
            np.testing.assert_allclose(part.own_n,n,rtol=0,atol=0)
        for prior_event,part in history.groupby('event'):
            old_features=read(run/'events'/str(int(prior_event))/'features.csv');cols=list(old_features.columns)
            a=part[cols].sort_values('case_id').reset_index(drop=True).copy();b=old_features.sort_values('case_id').reset_index(drop=True).copy()
            for col in ['event_type','band_group']:a[col]=a[col].map(str);b[col]=b[col].map(str)
            pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12,atol=1e-12)
        replay=replay_neural(directory/'models/neural_gate',features)
        for method,values in replay.items():
            expected=pred.query('method==@method').set_index('case_id').loc[features.case_id].predicted
            np.testing.assert_allclose(values,expected,rtol=1e-12,atol=1e-6)
        replayed+=1
        support.append({'event':event,'split':row['split'],'raw1500_training_cases_at_start':raw_audit['target_rank_support'].get('1500',0),
            'issued1500_cases':int((features['rank']==1500).sum()),'calibration1500_activities':int(history.loc[history['rank']==1500,'event'].nunique())})
    for split in ['validation','test']:
        p=read(run/f'{split}_predictions.csv');ids=set(data.case_index(split).case_id)
        for _,g in p.groupby('method'):assert set(g.case_id)==ids and not g.case_id.duplicated().any()
    assert selected['selected_recipe']==min(cfg['candidates'],key=lambda m:(selected['validation_scores'][m],cfg['candidates'].index(m)))
    labels=data.labels('validation');valid=read(run/'validation_predictions.csv')
    for method,value in selected['validation_scores'].items():
        s=score_temporal(data.case_index('validation'),data.memberships('validation'),labels,valid.query('method==@method'),stage_catalog=data.manifest['protocol']['stages'])
        assert abs(s['overall']['event_macro_mape_pct']-value)<1e-9
    pd.DataFrame(support).to_csv(records/'rolling_support.csv',index=False)
    receipt={'status':'pass','dataset_id':cfg['dataset_id'],'run_id':cfg['run_id'],'audited_events':len(events),'raw_models_replayed':raw_replayed,
        'neural_groups_replayed':replayed,'protected_old_files':len(protected),'selection_sha256':selection_sha,'blind_manifest_sha256':sha256(run/'blind_manifest.json'),
        'selected_method':selected['selected_recipe'],'validation_mape_pct':selected['validation_scores'][selected['selected_recipe']],
        'independent_test_label_file_read':False,'legally_released_past_test_errors_read':True,'audit_source_sha256':sha256(Path(__file__)),
        'parent_metadata_note':'Inherited V2 fit_policy/training_evaluation prose remains; V3 version, feedback/history_boundary and run-specific32+35scope govern.',
        'checks':['fixed panels','strict releases/seals','all frozen legal fitting cases','raw and neural checkpoint replay','event-equal weights',
                  'historical input/expert states immutable','cold rank own_n truthful','original2rank4controls identical','validation-only selection','old/production preservation']}
    write_new_json(records/'pre_scoring_audit.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=RECORDS);a=p.parse_args();audit(a.run,a.records)
