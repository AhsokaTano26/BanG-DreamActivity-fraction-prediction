"""Freeze cross-experiment choice using validation only, before new scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.rolling_dataset import RollingDataset
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.temporal_evaluate import score_temporal

RUNS=[ROOT/'analysis/rolling_results/v3_next_warm_20261116',ROOT/'analysis/rolling_results/v3_next_bases_20261116']


def freeze(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    entries=[]
    for run in RUNS:
        if (run/'evaluation_manifest.json').exists():raise ValueError('New independent scoring occurred before global freeze')
        seal=verify_seal(run/'blind_manifest.json');cfg=json.loads((run/'run_manifest.json').read_text())
        if any(sha256(ROOT/n)!=h for n,h in cfg['source_hashes'].items()):raise ValueError('Run source changed')
        selected=json.loads((run/'selection.json').read_text());method=selected['selected_recipe']
        if method not in selected['candidates'] or '_seed' in method:raise ValueError('Selection candidate is not preregistered')
        data=RollingDataset(seal['dataset_directory'])
        if data.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Dataset mismatch')
        frame=pd.read_csv(run/'validation_predictions.csv').query('method == @method')
        score=score_temporal(data.case_index('validation'),data.memberships('validation'),data.labels('validation'),frame)
        value=score['overall']['event_macro_mape_pct']
        if not np.isclose(value,selected['validation_scores'][method],atol=1e-9,rtol=0):raise ValueError('Validation choice score mismatch')
        entries.append({'run':str(run.relative_to(ROOT)),'run_id':seal['run_id'],'dataset_id':seal['dataset_id'],
            'selected_method':method,'validation_mape_pct':value,'selection_sha256':sha256(run/'selection.json'),
            'blind_manifest_sha256':sha256(run/'blind_manifest.json')})
    if len({r['dataset_id'] for r in entries})!=1:raise ValueError('Runs have different cases')
    winner=min(entries,key=lambda r:(r['validation_mape_pct'],RUNS.index(ROOT/r['run'])))
    write_new_json(output,{'rule':'minimum validation event-macro among the two preregistered run selections;warm-first tie',
        'runs':entries,'selected':winner,'new_test_score_read':False,'exposure':'Previously explored test, reused validation and retrospective metadata',
        'source_sha256':sha256(Path(__file__))})
    print(json.dumps({'selected':winner,'run_choices':entries},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);freeze(p.parse_args().output)
