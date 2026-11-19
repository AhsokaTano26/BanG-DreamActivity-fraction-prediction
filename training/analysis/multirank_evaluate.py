"""Independent one-shot scoring of sealed supplementary 19-rank predictions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256,write_json
from analysis.holdout_evaluate import score_predictions
from analysis.multirank_dataset import MultirankDataset,identity
from analysis.multirank_model import verify_run,metadata_for

def evaluate(run,output=None):
    run=Path(run);seal=verify_run(run);data=MultirankDataset(seal['dataset_directory']);output=Path(output or run)
    if data.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Dataset differs')
    sources=['training/analysis/multirank_evaluate.py','training/analysis/holdout_evaluate.py','training/analysis/multirank_dataset.py',
             'training/analysis/multirank_model.py','training/analysis/holdout_dataset.py','training/analysis/backtest.py','training/analysis/event_features.py']
    ident={'blind_manifest_sha256':sha256(run/'blind_test_manifest.json'),'dataset_id':data.manifest['dataset_id'],
        'sources':{n:sha256(ROOT/n) for n in sources},'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas']}}
    evaluation_id=identity(ident);receipt=output/'evaluation_manifest.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['evaluation_id']!=evaluation_id:raise FileExistsError('Changed evaluator; use separate --output')
        for n,h in saved['artifacts'].items():
            if sha256(output/n)!=h:raise ValueError('Scored artifact changed')
        return json.loads((output/'evaluation.json').read_text())
    if any((output/n).exists() for n in ['evaluation.json','test_summary.csv','evaluator_sources']):
        raise FileExistsError('Scoring output already exists')
    selection=json.loads((run/'selection.json').read_text());pred=pd.read_csv(run/'test_predictions.csv')
    if set(pred.method)!=set(selection['reported_methods']):raise ValueError('Method set mismatch')
    meta=metadata_for(data);cases=[]
    for s in data.inputs('test'):
        cases.append({'case_id':s['case_id'],'event':s['curve'].event,'rank':s['curve'].rank,'stage':s['stage'],
            'event_type':s['curve'].kind,'band_group':meta.band_only(s['curve'])['band_group']})
    labels=data.labels('test',allow_test=True)
    scores={m:score_predictions(cases,labels,g.to_dict('records')) for m,g in pred.groupby('method')}
    result={'evaluation_id':evaluation_id,'dataset_id':data.manifest['dataset_id'],'selection':selection['selected_recipe'],
        'selected':scores[selection['selected_recipe']]['overall'],'test':scores,'scope':data.manifest['protocol']['scope'],
        'exposure':data.manifest['protocol']['exposure'],'metadata_caveat':data.manifest['protocol']['metadata_caveat']}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'evaluation.json',result)
    pd.DataFrame([{'method':m,**s['overall']} for m,s in scores.items()]).to_csv(output/'test_summary.csv',index=False)
    directory=output/'evaluator_sources';directory.mkdir()
    for name in sources:shutil.copyfile(ROOT/name,directory/Path(name).name)
    files=['evaluation.json','test_summary.csv']+[str(p.relative_to(output)) for p in directory.iterdir()]
    write_json(receipt,{'evaluation_id':evaluation_id,'identity':ident,'artifacts':{n:sha256(output/n) for n in files}})
    print(json.dumps(result['selected']),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();evaluate(a.run,a.output)
