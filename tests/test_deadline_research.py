import importlib.util
from dataclasses import replace
import unittest
from unittest import mock
import numpy as np
from analysis.backtest import Curve, snapshot
from analysis.deadline_diagnostics import at_time


class DeadlineResearchTests(unittest.TestCase):
    def curve(self,event,start,rank=1000):
        times=start+np.arange(101)*3600000
        return Curve(event,rank,'live_try',start,start+100*3600000,times,100*np.arange(1,102,dtype=float)**1.1)

    def test_exact_observation_has_no_interpolation_span(self):
        curve=self.curve(1,1000000)
        self.assertEqual(at_time(curve,curve.times[40]),(curve.scores[40],0))
        _,span=at_time(curve,curve.times[40]+1800000)
        self.assertEqual(span,1)
        self.assertTrue(np.isnan(at_time(curve,curve.start-1)[0]))

    def test_history_excludes_current_future_and_unfinished_events(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.deadline_features'))
        from analysis.deadline_features import history_features
        past=self.curve(1,1000000);target=self.curve(2,past.end+86400000)
        future=self.curve(3,target.end+86400000);overlap=self.curve(4,target.start-3600000)
        sample=snapshot(target,target.start+75*3600000)
        expected=history_features([sample],[past])
        actual=history_features([sample],[past,target,future,overlap])
        np.testing.assert_allclose(expected.to_numpy(),actual.to_numpy(),equal_nan=True)
        changed=replace(target,scores=np.r_[target.scores[:76],target.scores[76:]*100])
        observed=snapshot(changed,sample['cutoff'])
        np.testing.assert_allclose(history_features([observed],[past]).to_numpy(),expected.to_numpy(),equal_nan=True)

    def test_history_units_and_missing_support(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.deadline_features'))
        from analysis.deadline_features import history_features,prior_offset
        past=self.curve(1,1000000);target=self.curve(2,past.end+86400000)
        sample=snapshot(target,target.start+75*3600000);a=history_features([sample],[past])
        scaled=snapshot(replace(target,scores=target.scores*100),sample['cutoff'])
        b=history_features([scaled],[replace(past,scores=past.scores*100)])
        np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),rtol=1e-10,atol=1e-10,equal_nan=True)
        missing=history_features([sample],[])
        self.assertEqual(missing['hist_count'].iloc[0],0)
        self.assertTrue(np.isnan(missing['hist_shape_logk'].iloc[0]))
        self.assertEqual(prior_offset(missing)[0],0)

    def test_unknown_velocity_is_not_a_perfect_shape_match(self):
        from analysis.deadline_features import history_features,prior_offset
        past=self.curve(1,1000000);target=self.curve(2,past.end+86400000)
        past=replace(past,times=past.times[[73,74,76,100]],scores=past.scores[[73,74,76,100]])
        sample=snapshot(target,target.start+75*3600000)
        frame=history_features([sample],[past])
        self.assertTrue(np.isfinite(frame['hist_deadline_logk'].iloc[0]))
        self.assertTrue(np.isnan(frame['hist_shape_logk'].iloc[0]))
        self.assertTrue(np.isnan(frame['hist_effective_n'].iloc[0]))
        self.assertEqual(frame['hist_count'].iloc[0],0)
        self.assertEqual(prior_offset(frame)[0],frame['hist_deadline_logk'].iloc[0])

    def test_development_never_reads_test_data_and_calibration_is_chronological(self):
        import json
        import pandas as pd
        from test_holdout_dataset import HoldoutDatasetTests
        from analysis import deadline_benchmark as b
        from analysis.holdout_dataset import sha256,write_json
        fixture=HoldoutDatasetTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        data=fixture.freeze();ref=fixture.root/'reference';ref.mkdir()
        validation=data.inputs('validation')
        pd.DataFrame(b.rows(validation,'reference',np.array([s['score']/s['progress'] for s in validation]))).to_csv(ref/'validation_predictions.csv',index=False)
        write_json(ref/'selection.json',{'selected_recipe':'reference'})
        write_json(ref/'blind_test_manifest.json',{'run_id':'synthetic','dataset_id':data.manifest['dataset_id'],
                   'artifacts':{p.name:sha256(p) for p in ref.iterdir()}})
        original_labels,original_inputs=data.labels,data.inputs
        def labels(split,*args,**kwargs):
            if split=='test':raise AssertionError('test answers read')
            return original_labels(split,*args,**kwargs)
        def inputs(split,*args,**kwargs):
            if split=='test':raise AssertionError('test inputs read')
            return original_inputs(split,*args,**kwargs)
        data.labels,data.inputs=labels,inputs
        run=fixture.root/'development'
        with mock.patch.multiple(b,FOLD_LEFT=[6,8,10,12],FOLD_SIZE=2,TREE_PARAMETERS=b.TREE_PARAMETERS|{'iterations':5}):
            b.develop(run,data,ref)
        folds=json.loads((run/'folds.json').read_text())
        for f in folds:
            self.assertFalse(set(f['calibration_events'])&set(f['held_events']))
            expected={e for previous in folds[:f['fold']] for e in previous['held_events']}
            self.assertEqual(set(f['calibration_events']),expected)
        results=pd.read_csv(run/'training_oof_predictions.csv');first=results[results.fold==0]
        base=first[first.method=='remaining_rank'].set_index('case_id').predicted
        corrected=first[first.method=='oof_correction_1.0'].set_index('case_id').predicted
        np.testing.assert_array_equal(base.sort_index(),corrected.sort_index())
        self.assertEqual(len(pd.read_csv(run/'validation_predictions.csv')),7*len(validation))
        with self.assertRaises(FileExistsError):b.develop(run,data,ref)


if __name__=='__main__':unittest.main()
