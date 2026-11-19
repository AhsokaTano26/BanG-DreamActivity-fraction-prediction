import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_adaptive_sharing import START, panel

try:
    replay = importlib.import_module('replay')
except ModuleNotFoundError:
    replay = None


def fixture():
    frames=[];queries=[]
    for index in range(6):
        start=START+index*20
        rows=panel(index+1,start,{1000:.2,2000:.2})
        rows=rows.drop(columns=['issued_ind','issued_pair','issued_adjacent','issued_samepanel'])
        rows['actual']=rows.base_prediction*np.exp(-rows.log_error)
        rows['expert__independent']=rows.base_prediction
        frames.append(rows)
        queries.append(dict(event=index+1,start_ms=start,end_ms=start+10,split='train'))
    cfg={'schedule':{'warmup':2,'selection':2,'diagnostic':2},
         'candidates':[{'name':'independent_fixed_recipe','kind':'fixed','lambda':0.,'pool':'pair'},
                       {'name':'fixed_half','kind':'fixed','lambda':.5,'pool':'pair'}],
         'budgets':{'wall_seconds':30}}
    return pd.concat(frames,ignore_index=True),queries,cfg


class CoordinatorTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(replay,'coordinator module is missing')

    def test_scores_mean_events_not_snapshots(self):
        frame=pd.DataFrame({'event':[1,1,2],'rank':[1000,2000,1000],
                            'prediction':[100.,200.,100.],'actual':[100.,100.,50.]})
        result=replay.metrics(frame)
        self.assertAlmostEqual(result['event_macro_mape'],75.)
        self.assertAlmostEqual(result['case_mape'],200/3)

    def test_selection_sealed_before_final_block_and_no_overwrite(self):
        records,queries,cfg=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            replay.run_replay(records,queries,cfg,output,allow_fixture=True)
            seal=json.loads((output/'selection_seal.json').read_text())
            self.assertEqual(seal['issued_events_before_seal'],4)
            self.assertEqual(seal['selected_candidate'],'independent_fixed_recipe')
            self.assertEqual(seal['selection_events'],[3,4])
            manifest=json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(manifest['predict_calls'],12)
            with self.assertRaises(FileExistsError):
                replay.run_replay(records,queries,cfg,output,allow_fixture=True)

    def test_late_outcome_perturbation_preserves_earlier_issue_and_selection(self):
        records,queries,cfg=fixture();perturbed=records.copy(deep=True)
        mask=perturbed.event>=5
        perturbed.loc[mask,'log_error']=-.7
        perturbed.loc[mask,'actual']=perturbed.loc[mask,'base_prediction']*np.exp(.7)
        with tempfile.TemporaryDirectory() as tmp:
            one=Path(tmp)/'one';two=Path(tmp)/'two'
            replay.run_replay(records,queries,cfg,one,allow_fixture=True)
            replay.run_replay(perturbed,queries,cfg,two,allow_fixture=True)
            for index in range(1,6):
                name=f'{index:02d}_event{index}.csv.gz'
                pd.testing.assert_frame_equal(pd.read_csv(one/'issued'/name),pd.read_csv(two/'issued'/name))
            a=json.loads((one/'selection_seal.json').read_text())
            b=json.loads((two/'selection_seal.json').read_text())
            self.assertEqual(a['selected_candidate'],b['selected_candidate'])
            self.assertEqual(a['scores'],b['scores'])

    def test_overlapping_selection_end_fails_closed_before_diagnosis(self):
        records,queries,cfg=fixture()
        records.loc[records.event==4,'end_ms']=queries[4]['start_ms']
        queries[3]['end_ms']=queries[4]['start_ms']
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with self.assertRaisesRegex(ValueError,'selection.*released'):
                replay.run_replay(records,queries,cfg,output,allow_fixture=True)
            self.assertFalse((output/'issued'/'05_event5.csv.gz').exists())

    def test_input_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x').write_text('changed')
            (root/'manifest.json').write_text(json.dumps({'files':{'x':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'SHA'):
                replay.verify_inputs(root)

    def test_metadata_categories_keep_string_identity(self):
        self.assertTrue(callable(getattr(replay,'load_train_vault',None)),
                        'explicit metadata-safe vault loader is missing')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.csv'
            path.write_text('event_type,band_group\n1,2\n')
            frame=replay.load_train_vault(path)
            self.assertEqual(frame.event_type.iloc[0],'1')
            self.assertEqual(frame.band_group.iloc[0],'2')


if __name__=='__main__':unittest.main()
