"""Freeze the suite overview using only each run's validation-selected method."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from analysis.holdout_dataset import ROOT, sha256, write_json
from analysis.current_benchmark import load_dataset
from analysis.holdout_evaluate import case_metadata, score_predictions
from analysis.temporal_evaluate import score_temporal
from analysis.dense_retrain_report import IDS, V3, EXPECTED_RUNS, check_receipt, require_inventory


def freeze(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    roots=[]
    for run in sorted((ROOT/'analysis/holdout_results').glob('dense_*_20261115')):
        if (run/'blind_test_manifest.json').exists():roots.append(('V1',run))
    for run in sorted((ROOT/'analysis/temporal_results').glob('dense_*_20261115')):
        if (run/'blind_manifest.json').exists():roots.append(('V2 19-rank' if run.name=='dense_19rank_20261115' else 'V2 two-rank',run))
    roots.append(('V3',V3))
    require_inventory([str(run.relative_to(ROOT)) for _,run in roots],EXPECTED_RUNS)
    actual=pd.Series([p for p,r in roots]).value_counts().to_dict()
    if actual!={'V1':7,'V2 two-rank':5,'V2 19-rank':1,'V3':1}:
        raise ValueError('Incomplete suite: '+str(actual))
    rows=[]
    for protocol,run in roots:
        sealpath=run/('blind_test_manifest.json' if protocol=='V1' else 'blind_manifest.json')
        seal=check_receipt(sealpath)
        if seal['dataset_id']!=IDS[protocol]:raise ValueError('Wrong cohort')
        if (run/'evaluation_manifest.json').exists():
            raise ValueError('New test scoring already occurred before suite selection')
        selection=json.loads((run/'selection.json').read_text())
        selected=selection.get('selected_recipe',selection.get('selected_on_validation'))
        data=load_dataset('v1' if protocol=='V1' else ('v3' if protocol=='V3' else 'v2'),seal['dataset_directory'])
        frame=pd.read_csv(run/'validation_predictions.csv')
        frame=frame[frame.method==selected]
        labels=data.labels('validation')
        if protocol=='V1':score=score_predictions(case_metadata(data.inputs('validation')),labels,frame.to_dict('records'))
        else:score=score_temporal(data.case_index('validation'),data.memberships('validation'),labels,frame)
        rows.append({'protocol':protocol,'run':str(run.relative_to(ROOT)),'selected_method':selected,
                     'validation_mape_pct':score['overall']['event_macro_mape_pct'],
                     'blind_seal_sha256':sha256(sealpath),'selection_sha256':sha256(run/'selection.json')})
    winners=[]
    for protocol in IDS:
        choices=[r for r in rows if r['protocol']==protocol]
        winners.append(min(choices,key=lambda r:(r['validation_mape_pct'],r['run'])))
    write_json(output,{'rule':'minimum validation event-macro MAPE among each preregistered runner selection; run path breaks ties',
                      'new_test_labels_read':False,'identical_v3_protocol_previously_scored':True,
                      'selected':winners,'runner_selections':rows,
                      'source_sha256':sha256(Path(__file__))})
    print(json.dumps(winners,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    freeze(p.parse_args().output)
