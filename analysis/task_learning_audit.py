"""Before-score causal, provenance and all-checkpoint replay audit; no label files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.multiline_dataset import MultilineDataset
from analysis.task_learning_runner import RUN,CONTROLS,CANDIDATES,validate_peer_binding
from analysis.task_peer_data import PeerCache
from analysis import task_learning_models as models
from analysis.task_learning_replay import replay as checkpoint_replay

RECORDS=ROOT/'analysis/task_learning_records/20261118'

def read(path):return pd.read_csv(path,float_precision='round_trip')

def same_rows(a,b,columns):
    a=a[columns].sort_values('case_id').reset_index(drop=True).copy()
    b=b[columns].sort_values('case_id').reset_index(drop=True).copy()
    for c in ['event_type','band_group']:
        if c in columns:a[c]=a[c].map(str);b[c]=b[c].map(str)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12,atol=1e-12)

def check_cache(cache):
    """Vectorized independent clocks/own-point checks for every cached link."""
    links=cache.arrays['links'];ages=cache.arrays['ages'];rows,cols=np.nonzero(links>=0)
    n=cache.nodes.iloc[links[rows,cols]];f=cache.cases.iloc[rows]
    assert np.array_equal(n.event,f.event)
    assert np.array_equal(n['rank'],np.asarray(cache.ranks)[cols])
    assert np.array_equal(n.start_ms,f.start_ms) and np.array_equal(n.end_ms,f.end_ms)
    lag=f.observed_at.to_numpy()-n.observed_at.to_numpy()
    assert (lag>=0).all() and (lag<=3600000).all()
    np.testing.assert_allclose(ages[rows,cols],lag/3600000,rtol=1e-6,atol=1e-7)
    owncol=np.array([cache.ranks.index(int(r)) for r in cache.cases['rank']])
    own=links[np.arange(len(links)),owncol];assert (own>=0).all()
    nodes=cache.nodes.iloc[own]
    assert np.array_equal(nodes.observed_at,cache.cases.observed_at)
    assert np.array_equal(nodes.prefix_points,cache.cases.prefix_points)
    assert not (models.FORBIDDEN|{'reference_log_error'})&set(cache.cases)
    return own,len(rows)

def audit(run=RUN,records=RECORDS):
    run,records=Path(run),Path(records)
    if (run/'evaluation_manifest.json').exists():raise ValueError('Run this audit before independent scoring')
    blind=verify_seal(run/'blind_manifest.json');cfg=json.loads((run/'run_manifest.json').read_text())
    prereg=json.loads((records/'preregistration.json').read_text())
    assert cfg==prereg['configuration'] and cfg['run_id']==blind['run_id']
    assert identity({k:v for k,v in cfg.items() if k!='run_id'})==cfg['run_id']
    assert cfg['candidates']==CANDIDATES and not cfg['allow_fixture']
    data=MultilineDataset(cfg['dataset_directory']);cache=PeerCache(cfg['peer_cache'])
    validate_peer_binding(data,cache,allow_fixture=False)
    own_nodes,peer_links=check_cache(cache)
    reference=Path(cfg['base_run']);refseal=verify_seal(reference/'blind_manifest.json')
    assert refseal['run_id']==cfg['reference_run_id'] and sha256(reference/'blind_manifest.json')==cfg['reference_blind_sha256']
    assert sha256(cache.root/'manifest.json')==cfg['peer_cache_manifest_sha256']
    for name,digest in cfg['source_hashes'].items():
        assert sha256(ROOT/name)==sha256(run/'sources'/name)==digest,name
    for name,digest in prereg['additional_sources'].items():assert sha256(ROOT/name)==digest,name
    protected=json.loads((records/'preservation_before.json').read_text())
    for name,digest in protected.items():assert sha256(name)==digest,name
    selection=json.loads((run/'selection.json').read_text());selection_sha=sha256(run/'selection.json')
    assert selection['candidates']==CANDIDATES and set(selection['validation_scores'])==set(CANDIDATES)
    assert selection['test_labels_used_for_selection'] is False
    assert selection['selected_recipe']==min(CANDIDATES,key=lambda m:(selection['validation_scores'][m],CANDIDATES.index(m)))
    queries={r['event']:r for r in data.event_index.values() if r['event'] in cfg['warm_forecast_events'] or r['split']!='train'}
    first_test=min(r['start_ms'] for r in queries.values() if r['split']=='test')
    assert selection['sealed_before_test_start_ms']==first_test
    states={};features={};event_predictions={};support=[];checkpoints=0;fitted=0;fallback=0;max_replay_error=0.
    for event,row in sorted(queries.items(),key=lambda p:(p[1]['start_ms'],p[0])):
        directory=run/'events'/str(event);seal=verify_seal(directory/'event_manifest.json')
        a=json.loads((directory/'audit.json').read_text());f=read(directory/'features.csv');h=read(directory/'released_residuals.csv')
        p=read(directory/'predictions.csv');state=json.loads((directory/'relation_state.json').read_text())
        assert seal['run_id']==cfg['run_id'] and seal['query_start_ms']==row['start_ms'] and seal['end_ms']==row['end_ms']
        expected={e for e,r in queries.items() if r['end_ms']<row['start_ms']}
        assert expected==set(h.event)=={r['event'] for r in a['released_forecasts']}
        assert a['training_rows']==len(h) and a['history_events']==len(expected) and a['current_outcome_released'] is False
        for r in a['released_forecasts']:
            assert all(r[k]==queries[r['event']][k] for k in ['split','start_ms','end_ms','sha256'])
            assert r['prediction_seal_sha256']==sha256(run/'events'/str(r['event'])/'event_manifest.json')
        assert state==models.relation_state(h,query_start_ms=row['start_ms'],ranks=cfg['ranks'])
        for e,part in h.groupby('event'):same_rows(part,features[int(e)],list(features[int(e)].columns))
        old=reference/'events'/str(event);verify_seal(old/'event_manifest.json')
        assert a['reference_event_manifest_sha256']==sha256(old/'event_manifest.json')
        # Old immutable feedback is an independent algebraic check of the anchor errors.
        oh=read(old/'released_residuals.csv').set_index('case_id').reindex(h.case_id)
        np.testing.assert_allclose(h.reference_log_error.to_numpy(float),
            np.log(h.reference_prediction.to_numpy(float)/oh.base_prediction.to_numpy(float))+oh.log_error.to_numpy(float),rtol=1e-10,atol=1e-12)
        oldpred=read(old/'predictions.csv');table=p.pivot(index='case_id',columns='method',values='predicted')
        for new,oldname in CONTROLS.items():
            original=oldpred[oldpred.method==oldname].set_index('case_id').loc[table.index].predicted
            np.testing.assert_array_equal(table[new],original)
        cases=data.case_index(row['split']).query('event==@event')
        assert not f.case_id.duplicated().any() and set(f.case_id)==set(cases.case_id)
        same_rows(f,cases,['case_id','event','rank','start_ms','end_ms','cutoff','observed_at','current'])
        assert set(p.method)==set(cfg['methods']) and len(p)==len(f)*len(cfg['methods'])
        for _,part in p.groupby('method'):assert set(part.case_id)==set(f.case_id) and not part.case_id.duplicated().any()
        assert (p.predicted>=p.current).all() and (p.predicted<=p.current*np.exp(4)+1e-6).all()
        if row['split']=='test':assert a['selection_sha256']==selection_sha
        if row['split']=='validation':assert selection['validation_event_seals'][str(event)]==sha256(directory/'event_manifest.json')
        mcfg=json.loads((directory/'models/config.json').read_text());states[event]=state
        assert mcfg['training_graph_hashes']=={str(e):identity(states[int(e)]) for e in h.event.unique()}
        assert mcfg['current_graph_sha256']==identity(state) and mcfg['case_ids']==f.case_id.tolist()
        assert mcfg['history_events']==len(expected) and mcfg['shrinkage']==len(expected)/(len(expected)+20)
        assert mcfg['recipe']==models.RECIPE and mcfg['epochs']==60 and mcfg['minimum_events']==12
        if len(expected)>=12:
            assert mcfg['status']=='fitted';fitted+=1
            prep=json.loads((directory/'models/preprocessor.json').read_text())
            assert sha256(directory/'models/preprocessor.json')==mcfg['preprocessor_sha256']
            weights=models.event_weights(h);indices=np.array([cache.positions[c] for c in h.case_id])
            nm,ns=models.weighted_scale(cache.arrays['values'][own_nodes[indices]],weights)
            _,_,issued_support=models.issue_features(h,states,cfg['ranks'])
            cm,cs=models.weighted_scale(models.context_numeric(h,issued_support),weights)
            for name,value in [('node_mean',nm),('node_std',ns),('context_mean',cm),('context_std',cs)]:
                np.testing.assert_array_equal(prep[name],value)
            assert prep['fit_case_ids']==h.case_id.tolist() and prep['fit_events']==sorted(expected)
            assert prep['categories']=={k:sorted(set(h[k].map(str))) for k in ['event_type','band_group']}
            saved=read(directory/'models/training_weights.csv')
            assert saved.case_id.tolist()==h.case_id.tolist()
            np.testing.assert_allclose(saved.weight,weights,rtol=1e-12,atol=1e-15)
            np.testing.assert_allclose(saved.groupby('event').weight.sum(),1.,rtol=1e-12)
            checkpoints+=len(models.MODES)*len(models.SEEDS)
        else:assert mcfg['status']=='reference_fallback';fallback+=1
        replay=checkpoint_replay(directory/'models',cache.batch(f.case_id),f,state)
        for method,values in replay.items():
            saved=table.loc[f.case_id,method].to_numpy()
            np.testing.assert_allclose(values,saved,rtol=1e-12,atol=1e-6)
            max_replay_error=max(max_replay_error,float(np.max(np.abs(values-saved))))
        for mode in models.MODES:
            np.testing.assert_array_equal(table[mode],table[[f'{mode}_seed{s}' for s in models.SEEDS]].mean(axis=1))
        support.append({'event':event,'split':row['split'],'history_events':len(expected),'network_status':mcfg['status'],
            'T1500_ended_forecasts':state['rank_counts'][cfg['ranks'].index(1500)],'T1500_current_cases':int((f['rank']==1500).sum())})
        features[event]=f;event_predictions[event]=p
        print(json.dumps({'audited_event':event,'status':mcfg['status'],'replayed_checkpoints_total':checkpoints}),flush=True)
    for split in ['train','validation','test']:
        p=read(run/('warm_predictions.csv' if split=='train' else split+'_predictions.csv'))
        wanted=pd.concat([event_predictions[e] for e,r in queries.items() if r['split']==split],ignore_index=True)
        same_rows(p.assign(case_id=p.case_id+':'+p.method),wanted.assign(case_id=wanted.case_id+':'+wanted.method),list(p.columns))
    # Recompute selection from outcomes already legally released at the first test start.
    first=min((r for r in queries.values() if r['split']=='test'),key=lambda r:(r['start_ms'],r['event']))
    h=read(run/'events'/str(first['event'])/'released_residuals.csv').set_index('case_id')
    val=read(run/'validation_predictions.csv')
    for method,score in selection['validation_scores'].items():
        p=val[val.method==method];truth=h.loc[p.case_id]
        actual=truth.reference_prediction.to_numpy()*np.exp(-truth.reference_log_error.to_numpy())
        errors=pd.DataFrame({'event':p.event.to_numpy(),'ape':np.abs(p.predicted.to_numpy()/actual-1)*100})
        np.testing.assert_allclose(errors.groupby('event').ape.mean().mean(),score,rtol=0,atol=1e-9)
    for name,digest in protected.items():assert sha256(name)==digest,name
    pd.DataFrame(support).to_csv(records/'rolling_support.csv',index=False)
    receipt={'status':'pass','run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],'audited_events':len(queries),
        'fitted_queries':fitted,'reference_fallback_queries':fallback,'seed_checkpoints_replayed':checkpoints,
        'all_seed_and_ensemble_predictions_replayed':len(queries)*16,'maximum_absolute_replay_error':max_replay_error,
        'cached_cases_clock_checked':len(cache.cases),'peer_links_clock_checked':peer_links,'protected_old_files':len(protected),
        'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'selection_sha256':selection_sha,
        'selected_method':selection['selected_recipe'],'independent_test_label_file_read':False,
        'legally_released_past_test_errors_read':True,'audit_source_sha256':sha256(Path(__file__)),
        'replay_reader_sha256':sha256(ROOT/'analysis/task_learning_replay.py'),
        'replay_layout':'event_type then band_group, matching frozen training order; no artifact mutation',
        'checks':['preregistered exact sources/cache/reference','all peer and own observation clocks','strict own sealed feedback',
          'original historical graph hashes','independent anchor error algebra','all 5 controls identical','every frozen case',
          'own-history-only weighted node/context preprocessing','equal activity training mass','all 12 seed checkpoint replay',
          'arithmetic final-score ensembles','validation-only eligible selection','all old/production files preserved']}
    write_new_json(records/'pre_scoring_audit.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--records',type=Path,default=RECORDS)
    a=p.parse_args();audit(a.run,a.records)
