"""Check that the disclosed budget extension changed only the epoch ceiling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.holdout_dataset import ROOT,HoldoutDataset,write_json,sha256
from analysis.quality_irregular_benchmark import verify_run


def main():
    root=ROOT/'analysis/quality_irregular_results';first=root/'v1';second=root/'v2';data=HoldoutDataset()
    for run in [first,second]:verify_run(run,data)
    a=json.loads((first/'run_manifest.json').read_text());b=json.loads((second/'run_manifest.json').read_text())
    expected=json.loads(json.dumps(a));expected['neural_settings']['epochs']=240
    for key in a:
        if key not in ['run_id','source_hashes']:assert expected[key]==b[key],key
    for name,digest in a['source_hashes'].items():assert b['source_hashes'][name]==digest
    assert set(b['source_hashes'])-set(a['source_hashes'])=={
        'analysis/quality_irregular_extension.py','docs/superpowers/plans/2026-11-11-irregular-convergence.md'}
    traces={};comparisons=[]
    for name in a['neural_recipes']:
        for seed in a['seeds']:
            x=json.loads((first/'models'/f'{name}_{seed}.json').read_text())
            y=json.loads((second/'models'/f'{name}_{seed}.json').read_text())
            assert y['trace'][:len(x['trace'])]==x['trace'],(name,seed)
            traces[f'{name}_{seed}']=len(x['trace'])
            comparisons.append({'method':name,'seed':seed,'epochs_80':x['epochs'],'best_epoch_80':x['best_epoch'],
                'validation_mape_80':x['validation_mape'],'epochs_240':y['epochs'],'best_epoch_240':y['best_epoch'],
                'validation_mape_240':y['validation_mape']})
    for filename in ['training_oof_predictions.csv','training_predictions.csv.gz','validation_predictions.csv']:
        x=pd.read_csv(first/filename);y=pd.read_csv(second/filename)
        names=a['statistical_recipes']
        x=x[x.method.isin(names)].sort_values(['method','case_id']).reset_index(drop=True)
        y=y[y.method.isin(names)].sort_values(['method','case_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(x,y,check_exact=True)
    pd.DataFrame(comparisons).to_csv(root/'budget_comparison.csv',index=False)
    write_json(root/'budget_verification.json',{'first_run_id':a['run_id'],'extended_run_id':b['run_id'],
        'identical_initial_epoch_traces':traces,'all_statistical_control_predictions_identical':True,
        'only_maximum_epochs_changed':True,'test_inputs_or_answers_requested':False,'source_sha256':sha256(Path(__file__))})
    print('All initial training traces and statistical predictions match exactly; only epoch ceiling changed.')


if __name__=='__main__':main()
