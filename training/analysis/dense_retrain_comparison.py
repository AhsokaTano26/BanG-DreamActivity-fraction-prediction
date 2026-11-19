"""Post-seal, activity-paired comparisons for validation-selected dense methods."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from analysis.current_benchmark import load_dataset
from analysis.holdout_dataset import ROOT, sha256, write_json
from analysis.holdout_evaluate import case_metadata, score_predictions
from analysis.dense_retrain_report import (V3, EXPECTED_RUNS, check_receipt,
    require_inventory, require_selection_binding, require_evaluation_binding, require_suite_choices)


def ordered_test_events(partitions, events):
    parts=partitions.loc[partitions['split']=='test'].sort_values(['start_ms','event']).drop_duplicates('event')
    order=[int(e) for e in parts.event if int(e) in events]
    if set(order)!=set(events):raise ValueError('Unknown test activity membership')
    return order


def compare(a, b, protocol):
    run_a, method_a = a; run_b, method_b = b
    name='blind_test_manifest.json' if protocol=='v1' else 'blind_manifest.json'
    seal_a=check_receipt(run_a/name);seal_b=check_receipt(run_b/name)
    audit=[]
    for run,seal in [(run_a,seal_a),(run_b,seal_b)]:
        receipt=check_receipt(run/'evaluation_manifest.json')
        evaluation=json.loads((run/'evaluation.json').read_text())
        require_evaluation_binding(receipt,evaluation,seal,sha256(run/name))
        audit.append({'run':str(run.relative_to(ROOT)),'blind_manifest_sha256':sha256(run/name),
                      'evaluation_manifest_sha256':sha256(run/'evaluation_manifest.json'),
                      'evaluation_id':receipt['evaluation_id']})
    data=load_dataset(protocol,seal_a['dataset_directory'])
    # V2 and V3 may share cases under different information rules, but this
    # comparison deliberately uses one identical protocol/dataset per pair.
    if seal_a['dataset_id']!=seal_b['dataset_id']:raise ValueError('Different comparison cohort or protocol')
    labels=data.labels('test',allow_test=True)
    cases=case_metadata(data.inputs('test'))
    errors=[]
    for run,method in [a,b]:
        pred=pd.read_csv(run/'test_predictions.csv');pred=pred[pred.method==method]
        score_predictions(cases,labels,pred.to_dict('records'))
        frame=pd.DataFrame(cases).merge(pred[['case_id','predicted']],on='case_id',validate='one_to_one')
        frame['ape']=np.abs(frame.predicted/frame.case_id.map(labels)-1)*100
        errors.append(frame.groupby('event').ape.mean())
    parts=pd.read_csv(data.root/'partitions.csv')
    order=ordered_test_events(parts,set(errors[0].index))
    if set(errors[0].index)!=set(errors[1].index):raise ValueError('Activity panel differs')
    delta=(errors[0].loc[order]-errors[1].loc[order]).to_numpy();n=len(delta)
    result={'protocol':protocol,'candidate_run':str(run_a.relative_to(ROOT)),'candidate_method':method_a,
            'reference_run':str(run_b.relative_to(ROOT)),'reference_method':method_b,
            'events':n,'candidate_minus_reference_mape_points':float(delta.mean()),
            'event_ids_in_time_order':order,'event_differences':delta.tolist(),'improved_events':int((delta<0).sum()),
            'audit':audit}
    rng=np.random.default_rng(42);draws=5000
    result['iid_event_bootstrap_95_ci']=np.quantile(delta[rng.integers(0,n,(draws,n))].mean(axis=1),[.025,.975]).tolist()
    result['circular_block_bootstrap_95_ci']={}
    for size in [3,5]:
        rng=np.random.default_rng(42)
        starts=rng.integers(0,n,(draws,int(np.ceil(n/size))))
        indices=((starts[:,:,None]+np.arange(size))%n).reshape(draws,-1)[:,:n]
        result['circular_block_bootstrap_95_ci'][str(size)]=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()
    return result


def main(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    suitepath=ROOT/'analysis/dense_retrain_records/20261115/pretest_selection.json'
    suite=json.loads(suitepath.read_text())
    require_suite_choices(suite)
    require_inventory([r['run'] for r in suite['runner_selections']],EXPECTED_RUNS)
    for entry in suite['runner_selections']:
        run=ROOT/entry['run'];selection=json.loads((run/'selection.json').read_text())
        selected=selection.get('selected_recipe',selection.get('selected_on_validation'))
        require_selection_binding(run,run/('blind_test_manifest.json' if entry['protocol']=='V1' else 'blind_manifest.json'),selected,entry)
    for winner in suite['selected']:
        if winner not in suite['runner_selections']:raise ValueError('Suite winner changed')
    winners={r['protocol']:(ROOT/r['run'],r['selected_method']) for r in suite['selected']}
    roots={Path(r['run']).name:(ROOT/r['run'],r['selected_method']) for r in suite['runner_selections']}
    comparisons=[]
    for protocol,kind,ref in [('V1','v1','dense_core_20261115'),('V2 two-rank','v2','dense_base_20261115')]:
        comparisons.append(compare(winners[protocol],roots[ref],kind))
    run=V3
    comparisons.append(compare(winners['V3'],(run,'static_context'),'v3'))
    from analysis.innovation_common import local_sources
    source_hashes={name:sha256(ROOT/name) for name in local_sources([Path(__file__)])}
    write_json(output,{'draws':5000,'seed':42,'unit':'whole activity, paired; block sizes 3 and 5 fixed before test scoring',
                       'limitations':'Only 16 test activities; explored cohort and retrospective metadata. Intervals describe this panel, not an untouched future population.',
                       'comparisons':comparisons,'source_hashes':source_hashes,'suite_sha256':sha256(suitepath),
                       'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas']}})
    print(json.dumps(comparisons,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
