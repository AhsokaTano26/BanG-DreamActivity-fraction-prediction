import importlib.util,os,subprocess,sys,unittest
from pathlib import Path


class StructuredPipelineTests(unittest.TestCase):
    def test_checkpoint_selection_prediction_is_blind_and_sealed(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.structured_neural_benchmark'))
        script=r'''
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('tests').resolve()))
from test_holdout_dataset import HoldoutDatasetTests
from analysis import structured_neural_benchmark as b
from analysis.holdout_dataset import write_json,sha256
from analysis.holdout_evaluate import evaluate_run
fixture=HoldoutDatasetTests();fixture.setUp()
try:
    data=fixture.freeze();ref=fixture.root/'reference';ref.mkdir()
    method='synthetic_reference'
    for split in ['validation','test']:
        samples=data.inputs(split)
        pd.DataFrame(b.prediction_rows(samples,method,np.array([s['score']/s['progress'] for s in samples]))).to_csv(ref/f'{split}_predictions.csv',index=False)
    write_json(ref/'selection.json',{'selected_recipe':method})
    write_json(ref/'blind_test_manifest.json',{'run_id':'synthetic','dataset_id':data.manifest['dataset_id'],
        'artifacts':{p.name:sha256(p) for p in ref.iterdir() if p.is_file()}})
    b.RECIPES={n:b.RECIPES[n] for n in ['rank_mlp','dual_tcn_thin','cross_patch']}
    b.SEEDS=[42]
    original=b.HoldoutDataset.labels
    def guarded(self,split,**kwargs):
        if split=='test':raise AssertionError('model pipeline read test labels')
        return original(self,split,**kwargs)
    b.HoldoutDataset.labels=guarded
    args=argparse.Namespace(run=fixture.root/'run',dataset=data.root,reference=ref,device='cpu',epochs=2,batch_size=256,resume=False)
    b.develop(args,data)
    args.resume=True
    b.develop(args,data)
    config=json.loads((args.run/'run_manifest.json').read_text())
    for field in ['software','source_hashes','reference']:
        changed=copy.deepcopy(config)
        key='run_id' if field=='reference' else next(iter(changed[field]))
        changed[field][key]='invalid'
        try:b.verify_config(changed,data)
        except ValueError:pass
        else:raise AssertionError('identity drift accepted: '+field)
    prep=args.run/'preprocessor.joblib';saved=prep.read_bytes();prep.write_bytes(saved+b'changed')
    try:b.predict(args,data)
    except ValueError:pass
    else:raise AssertionError('modified preprocessor accepted')
    prep.write_bytes(saved)
    b.predict(args,data)
    before=(args.run/'test_predictions.csv').read_bytes()
    try:b.predict(args,data)
    except FileExistsError:pass
    else:raise AssertionError('sealed predictions overwritten')
    assert (args.run/'test_predictions.csv').read_bytes()==before
    b.HoldoutDataset.labels=original
    result=evaluate_run(args.run,data.root)
    assert result['test']['reference']['overall']['events']==3
    assert all(x['overall']['cases']==30 for x in result['test'].values())
finally:fixture.doCleanups()
'''
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/libomp/lib'
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],env=env,
                              capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout[-3000:]+'\n'+result.stderr[-4000:])


if __name__=='__main__':unittest.main()
