import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from analysis.backtest import Curve, snapshot

H = 3600000


class TemporalPairedTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_paired_features'),
                             'As-of paired feature builder required')
        from analysis import temporal_paired_features
        return temporal_paired_features

    def sample(self, rank, last, cutoff=5, event=1):
        c = Curve(event, rank, 'normal', 1, 100*H+1,
                  np.arange(last+1)*H+1, (np.arange(last+1)+1.)*(100 if rank == 1000 else 50))
        s = snapshot(c, cutoff*H+1)
        s.update(case_id=f'train:{event}:{rank}:{cutoff}', stage='fixture')
        return s

    def test_peer_future_suffix_cannot_change_primary_features(self):
        m = self.module()
        own, peer = self.sample(1000, 4), self.sample(2000, 5)
        before = m.prepare_paired([own, peer])
        changed = peer['curve'].scores.copy(); changed[-1] = 999999
        updated = snapshot(replace(peer['curve'], scores=changed), peer['cutoff'])
        updated.update(case_id=peer['case_id'], stage='fixture')
        after = m.prepare_paired([own, updated])
        pd.testing.assert_series_equal(before['paired'].iloc[0], after['paired'].iloc[0])
        self.assertEqual(before['audit'].iloc[0].peer_observed_at, 4*H+1)
        self.assertAlmostEqual(before['paired'].iloc[0].pair_log_score_ratio, np.log(2))
        self.assertEqual(before['paired'].iloc[0].peer_available, 1)

    def test_missing_or_later_origin_peer_never_drops_or_fills_a_case(self):
        m = self.module()
        own = self.sample(1000, 4)
        later = self.sample(2000, 6, cutoff=6)
        other_event = self.sample(2000, 5, event=2)
        arrays = m.prepare_paired([own, later, other_event])
        self.assertEqual(len(arrays['cases']), 3)
        self.assertTrue((arrays['paired'].peer_available == 0).all())
        sequence = m.sequence_arrays(arrays['paired'])
        self.assertEqual(sequence.shape, (3, 2, 33, 2))
        self.assertTrue(np.isfinite(sequence).all())
        np.testing.assert_array_equal(sequence[:, 1], 0)
        self.assertEqual(sequence[0, 0, -1, 0], 1)
        with self.assertRaises(ValueError):
            m.prepare_paired([own, own])
        mixed = dict(later, case_id='validation:1:2000:6')
        with self.assertRaises(ValueError):
            m.prepare_paired([own, mixed])

    def test_sequence_contains_exactly_the_flat_curve_values_and_masks(self):
        m = self.module()
        arrays = m.prepare_paired([self.sample(1000, 4), self.sample(2000, 5)])
        sequence = m.sequence_arrays(arrays['paired'])
        for branch, prefix in enumerate(['', 'peer__']):
            values = arrays['paired'][[f'{prefix}prefix_{i:02}' for i in range(33)]].fillna(0).to_numpy()
            masks = arrays['paired'][[f'{prefix}present_{i:02}' for i in range(33)]].to_numpy()
            np.testing.assert_allclose(sequence[:, branch, :, 0], values)
            np.testing.assert_allclose(sequence[:, branch, :, 1], masks)

    def test_empty_or_short_peer_after_truncation_remains_missing(self):
        m = self.module(); own = self.sample(1000, 4)
        for hours in [[4.25, 4.5, 4.75, 5], [3.75, 4.25, 4.75, 5]]:
            with self.subTest(hours=hours):
                c = Curve(1, 2000, 'normal', 1, 100*H+1,
                          (np.array(hours)*H+1).astype(np.int64), np.arange(1., 5.)*50)
                peer = snapshot(c, 5*H+1)
                peer.update(case_id='train:1:2000:5', stage='fixture')
                arrays = m.prepare_paired([own, peer])
                self.assertEqual(len(arrays['cases']), 2)
                self.assertEqual(arrays['paired'].iloc[0].peer_available, 0)
                self.assertEqual(arrays['audit'].iloc[0].peer_points, 0)
                self.assertTrue(np.isfinite(m.sequence_arrays(arrays['paired'])).all())

    def test_phase_weights_equalize_available_phases_and_event_totals(self):
        m = self.module()
        weights = m.phase_weights([1, 1, 1, 1, 2], [.1, .2, .8, .98, .6])
        self.assertAlmostEqual(weights[:4].sum(), weights[4])
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertAlmostEqual(weights[2], weights[3])
        self.assertAlmostEqual(weights.mean(), 1)


class TemporalPairedPipelineTests(unittest.TestCase):
    def test_network_can_use_the_early_part_of_both_prefixes(self):
        script = r'''
import torch
from analysis.temporal_paired_network import DualTCN
torch.set_num_threads(2)
model=DualTCN(3);model.eval()
with torch.no_grad():
 for p in model.parameters():p.fill_(.05)
sequence=torch.ones(1,2,33,2);tab=torch.zeros(1,3);rank=torch.zeros(1,dtype=torch.long)
before=model(sequence,tab,rank)
for branch in [0,1]:
 changed=sequence.clone();changed[0,branch,0,0]=5
 assert not torch.allclose(before,model(changed,tab,rank)), 'early prefix is outside effective receptive field'
'''
        env = os.environ.copy(); env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib'
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                                env=env, cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_three_seed_pipeline_is_blind_replayable_and_sealed(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_paired_benchmark'),
                             'Paired experiment entry point required')
        script = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('tests').resolve()))
import test_temporal_training as fixtures
from analysis.temporal_rank_subset import build_subset
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_train import run_training
from analysis.temporal_evaluate import verify_predictions,evaluate
from analysis.temporal_paired_benchmark import run_experiment,predict_saved
helper=fixtures.TemporalTrainingTests()
try:
 root,parent=helper.fixture();build_subset(parent,root/'subset');data=TemporalDataset(root/'subset')
 run_training(data,root/'reference',iterations=3)
 original=TemporalDataset.labels
 def guarded(self,split,**kw):
  if split=='test':raise AssertionError('test answers forbidden')
  return original(self,split,**kw)
 with patch.object(TemporalDataset,'labels',guarded),patch('sqlite3.connect',side_effect=AssertionError('SQLite forbidden')):
  selected=run_experiment(data,root/'reference',root/'paired',iterations=3,max_epochs=2)
 seal=verify_predictions(root/'paired')
 assert seal['reference_method']==selected['selected_recipe']
 assert selected['selected_recipe']==min(selected['candidates'],key=lambda m:selected['validation'][m]['event_macro_mape_pct'])
 assert all('seed' not in m for m in selected['candidates'])
 pred=pd.read_csv(root/'paired/test_predictions.csv').pivot(index='case_id',columns='method',values='predicted')
 seeds=['dual_tcn_seed42','dual_tcn_seed123','dual_tcn_seed2026']
 np.testing.assert_allclose(pred.dual_tcn_mean,pred[seeds].mean(axis=1))
 assert not pred.isna().any().any()
 for name in seeds+['catboost_paired','catboost_phase']:
  replay=predict_saved(root/'paired',list(data.inputs('test')),name)
  np.testing.assert_allclose(replay.predicted,replay.case_id.map(pred[name]),rtol=1e-6)
 weights=pd.read_csv(root/'paired/training_weights.csv')
 for name in ['event_weight','phase_weight']:
  totals=weights.groupby('event')[name].sum();assert np.allclose(totals,totals.iloc[0])
 scored=evaluate(root/'paired');assert len(scored['splits']['test'])==9
 try:run_experiment(data,root/'reference',root/'paired',iterations=3,max_epochs=2)
 except FileExistsError:pass
 else:raise AssertionError('sealed run overwritten')
 model=root/'paired/models/dual_tcn_seed42.pt';model.write_bytes(model.read_bytes()+b'x')
 try:verify_predictions(root/'paired')
 except ValueError:pass
 else:raise AssertionError('modified checkpoint accepted')
finally:helper.doCleanups()
'''
        env = os.environ.copy(); env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib'
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                                env=env, cwd=Path(__file__).resolve().parents[1], timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:]+'\n'+result.stderr[-5000:])
