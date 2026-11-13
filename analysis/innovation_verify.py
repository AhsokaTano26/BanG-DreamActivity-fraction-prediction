"""Replay sealed models and verify chronology/weights using blind inputs only."""
from pathlib import Path
import argparse
import importlib
import json
import joblib
import numpy as np
import pandas as pd
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.temporal_dataset import TemporalDataset
from analysis.innovation_common import prepare
from analysis.innovation_compare import verify_run


def verify(run):
    run=Path(run);out=run/'verification'
    if out.exists():raise FileExistsError(out)
    cfg=json.loads((run/'run_manifest.json').read_text())
    seal_name='blind_test_manifest.json' if cfg['protocol']=='v1' else 'blind_manifest.json'
    verify_run(run,seal_name)
    for name,digest in cfg['source_hashes'].items():
        if sha256(Path(name))!=digest:raise ValueError('Current model source differs: '+name)
    data=HoldoutDataset(cfg['dataset_directory']) if cfg['protocol']=='v1' else TemporalDataset(cfg['dataset_directory'])
    metadata=joblib.load(run/'models/metadata.joblib')
    train=prepare(data.inputs('train'),metadata);test=prepare(data.inputs('test'),metadata)
    train_by_id={c['case_id']:c for c in train['cases']}
    history=list(data.train_curves());times={c.event:(c.start,c.end) for c in history}
    predicted=pd.read_csv(run/'test_predictions.csv').pivot(index='case_id',columns='method',values='predicted')
    ids=[c['case_id'] for c in test['cases']];assert set(predicted.index)==set(ids)
    weights=pd.read_csv(run/'training_weights.csv')
    assert set(weights.case_id)==set(train_by_id)
    totals=weights.groupby('event').weight.sum();np.testing.assert_allclose(totals,totals.iloc[0])
    checked=[];inner_folds=0
    for name,spec in cfg['recipes'].items():
        module=importlib.import_module('analysis.innovation_'+spec['family'])
        bundle=joblib.load(run/f'models/{name}.joblib')
        p=module.predict_recipe(bundle,test)
        np.testing.assert_allclose(p,predicted.loc[ids,name].to_numpy(),rtol=1e-10,atol=1e-7)
        if hasattr(module,'predict_members'):
            members=module.predict_members(bundle,test)
            np.testing.assert_allclose(np.mean(list(members.values()),axis=0),p,rtol=1e-12)
            assert set(members)=={'seed_42','seed_123','seed_2026'}
            for member,value in members.items():
                np.testing.assert_allclose(value,predicted.loc[ids,name+'__'+member],rtol=1e-10)
        for fold in bundle.get('oof_folds',[]):
            if fold['train_events']:
                assert max(times[e][1] for e in fold['train_events'])<min(times[e][0] for e in fold['heldout_events'])
            assert set(fold['train_events']).isdisjoint(fold['heldout_events']);inner_folds+=1
        checked.append({'method':name,'cases_replayed':len(ids),'parameters':bundle.get('parameters')})
    folds=json.loads((run/'cv/folds.json').read_text())
    for fold in folds:
        fit=set(fold['training_events']);held=set(fold['heldout_events'])
        assert fit.isdisjoint(held)
        assert max(times[e][1] for e in fit)<min(times[e][0] for e in held)
        expected={c['case_id'] for c in train['cases'] if c['event'] in fit}
        p=pd.read_csv(run/f"cv/fold_{fold['fold']}/predictions.csv")
        test_ids={c['case_id'] for c in train['cases'] if c['event'] in held}
        for _,g in p.groupby('method'):assert g.case_id.is_unique and set(g.case_id)==test_ids
        for name,spec in cfg['recipes'].items():
            bundle=joblib.load(run/f"cv/fold_{fold['fold']}/{name}.joblib")
            found=bundle.get('training_case_ids',bundle.get('train_case_ids'))
            assert set(found)==expected
            if 'history' in bundle:
                assert {c.event for c in bundle['history']}==fit
    selection=json.loads((run/'selection.json').read_text())
    for name,spec in selection['combination_specs'].items():
        np.testing.assert_allclose(predicted.loc[ids,spec['members']].mean(axis=1),predicted.loc[ids,name],rtol=1e-10)
    out.mkdir()
    from analysis.innovation_retrieval import retrieval_features
    bundle=joblib.load(run/'models/retrieval__retrieval_features.joblib')
    coverage=[]
    for split in ['train','validation','test']:
        a=train if split=='train' else test if split=='test' else prepare(data.inputs(split),metadata)
        features=retrieval_features(a,bundle['history'])
        audit=pd.DataFrame(a['cases'])
        for key in ['retrieval_fallback','retrieval_eligible_count','retrieval_neighbor_count','retrieval_lag_mean_minutes']:
            audit[key]=features[key]
        audit.to_csv(out/f'{split}_retrieval_coverage.csv',index=False)
        coverage.append({'split':split,'cases':len(audit),'events':audit.event.nunique(),
            'fallback_case_fraction':float(audit.retrieval_fallback.mean()),
            'fallback_event_fraction':float(audit.groupby('event').retrieval_fallback.mean().mean())})
    result={'dataset_id':cfg['dataset_id'],'source_sha256':sha256(Path(__file__)),
        'model_replay':checked,'outer_folds_checked':len(folds),'inner_calibration_folds_checked':inner_folds,
        'equal_training_event_weights':True,'training_events':len(totals),'all_test_cases':len(ids),
        'retrieval_coverage':coverage,'answer_files_parsed':[],'status':'passed'}
    write_json(out/'verification.json',result)
    write_json(out/'manifest.json',{'files':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run)
