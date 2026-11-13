"""Recompute research metrics without requesting test inputs or answers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from collections import defaultdict
import json
import statistics
import pandas as pd
from analysis.holdout_dataset import HoldoutDataset,sha256,write_json
from analysis.holdout_evaluate import case_metadata,score_predictions
from analysis.event_features import EventFeatures


def main():
    root=Path(__file__).resolve().parent;run=root/'prototypes_v2';data=HoldoutDataset();meta=EventFeatures.load(data.metadata_directory)
    for path,sealname in [(run,'development_seal.json'),(root/'diagnostics_v1','manifest.json'),(root/'sampling_v1','manifest.json')]:
        seal=json.loads((path/sealname).read_text())
        for name,digest in seal['artifacts'].items():assert sha256(path/name)==digest,(path,name)
    config=json.loads((run/'run_manifest.json').read_text())
    for name,digest in config['source_hashes'].items():assert sha256(ROOT/name)==digest
    folds=json.loads((run/'folds.json').read_text());held={e for fold in folds for e in fold['held_events']}
    result={};differences=[]
    for split,filename in [('train','training_oof_predictions.csv'),('validation','validation_predictions.csv')]:
        samples=data.inputs(split);labels=data.labels(split)
        if split=='train':samples=[s for s in samples if s['curve'].event in held]
        cases=case_metadata(samples,meta);labels={s['case_id']:labels[s['case_id']] for s in samples}
        case_lookup={s['case_id']:s for s in samples};frame=pd.read_csv(run/filename);scores={}
        for name,group in frame.groupby('method'):
            for row in group.itertuples():
                s=case_lookup[row.case_id]
                assert row.event==s['curve'].event and row.rank==s['curve'].rank and row.predicted>=s['score']
                if split=='train':assert row.actual==labels[row.case_id]
            scores[name]=score_predictions(cases,labels,group.to_dict('records'))
            grouped=defaultdict(list)
            for row in group.itertuples():grouped[case_lookup[row.case_id]['curve'].event].append(abs(row.predicted/labels[row.case_id]-1)*100)
            independent=statistics.mean(statistics.mean(values) for values in grouped.values())
            difference=abs(independent-scores[name]['overall']['event_macro_mape_pct']);assert difference<1e-10;differences.append(difference)
        result[split]=scores
    # Exact replay of the preexisting baseline is a useful control for the new runner.
    previous=pd.read_csv(ROOT/'analysis/holdout_results/relationships_v3/validation_predictions.csv')
    previous=previous[previous.method=='catboost_remaining_invariant_rank'].set_index('case_id').predicted.sort_index()
    current=pd.read_csv(run/'validation_predictions.csv');current=current[current.method=='remaining_rank'].set_index('case_id').predicted.sort_index()
    assert previous.index.equals(current.index) and (previous==current).all()
    write_json(run/'metric_details.json',result)
    write_json(root/'verification.json',{'dataset_id':data.manifest['dataset_id'],'run_id':config['run_id'],
                'all_seals_valid':True,'source_hashes_valid':True,'baseline_validation_replay_exact':True,
                'independent_metric_max_difference':max(differences),'train_oof_events':len(held),
                'train_oof_cases':result['train']['remaining_rank']['overall']['cases'],
                'validation_events':result['validation']['remaining_rank']['overall']['events'],
                'validation_cases':result['validation']['remaining_rank']['overall']['cases'],
                'test_inputs_or_labels_requested':False,'source_sha256':sha256(Path(__file__)),
                'database_sha256':sha256(ROOT/'data/db.sqlite3')})
    print('All sources, artifacts, case identities and metrics verified; baseline replays exactly; no test data read.')


if __name__=='__main__':main()
