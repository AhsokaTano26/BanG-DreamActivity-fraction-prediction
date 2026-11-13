"""Run torch/tree integration in a subprocess to isolate macOS OpenMP runtimes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class HoldoutPipelineTests(unittest.TestCase):
    def test_training_is_blind_and_saved_models_can_be_scored(self):
        script = r'''
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('tests').resolve()))
from test_holdout_dataset import HoldoutDatasetTests
from analysis import holdout_benchmark as b
from analysis.holdout_evaluate import evaluate_run
fixture=HoldoutDatasetTests();fixture.setUp()
try:
    data=fixture.freeze();run=fixture.root/'run'
    original=b.HoldoutDataset.labels
    def blind_labels(self,split,**kwargs):
        if split=='test':raise AssertionError('training requested test labels')
        return original(self,split,**kwargs)
    b.HoldoutDataset.labels=blind_labels
    b.TREE_RECIPES={'ridge_growth':('baseline','ridge'),'catboost_peer':('peer','catboost')}
    b.NEURAL_RECIPES={'mlp':('baseline','mlp'),'tcn_peer':('peer','tcn')}
    b.run_benchmark(argparse.Namespace(dataset=data.root,run=run,device='cpu',max_epochs=1,batch_size=256,resume=False))
    import pandas as pd
    predicted=pd.read_csv(run/'test_predictions.csv')
    assert 'actual' not in predicted and 'target' not in predicted
    assert (run/'blind_test_manifest.json').exists()
    assert (run/'models/mlp_seed42.pt').exists()
    b.HoldoutDataset.labels=original
    result=evaluate_run(run,data.root)
    assert result['selection']['test_labels_used_for_selection'] is False
    assert result['test']['mlp_ensemble3']['overall']['events']==3
    assert 'evaluation_id' in result
    from analysis import holdout_evaluate as e
    before=(run/'evaluation.json').read_bytes()
    b.HoldoutDataset.labels=blind_labels
    cached=evaluate_run(run,data.root)
    assert cached['evaluation_id']==result['evaluation_id']
    assert (run/'evaluation.json').read_bytes()==before
    b.HoldoutDataset.labels=original
    old_identity=e.scoring_identity
    e.scoring_identity=lambda *a,**k:old_identity(*a,**k)|{'changed_scoring_implementation':True}
    try:evaluate_run(run,data.root)
    except FileExistsError as err:assert 'scor' in str(err)
    else:raise AssertionError('changed evaluator silently overwrote previous scores')
    e.scoring_identity=old_identity
    # Sealed checkpoints/preprocessors must detect later mutation.
    path=run/'models/preprocessor_peer.joblib'
    with path.open('ab') as f:f.write(b'changed')
    try:evaluate_run(run,data.root)
    except ValueError as e:assert 'hash' in str(e)
    else:raise AssertionError('tampered model accepted')
finally:fixture.doCleanups()
'''
        env = os.environ.copy()
        env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib'
        result = subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                                env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout[-4000:]+'\n'+result.stderr[-4000:])


if __name__ == '__main__':unittest.main()
