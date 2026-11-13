"""Temporal V2 must change forecast origins without leaking or moving activities."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from analysis.backtest import Curve
from analysis.multirank_dataset import build_extension,MultirankDataset
import test_multirank as fixtures
H=3600000

class TemporalProtocolTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_dataset'),'Temporal dataset missing')
        from analysis import temporal_dataset
        return temporal_dataset

    def fixture(self):
        helper=fixtures.MultirankTests();self.addCleanup(helper.doCleanups)
        root,old,extra=helper.fixture();build_extension(root/'ranks',old,extra,source_identity={'fixture':True})
        all_curves=[replace(c,rank=rank) for c in extra for rank in (10,1000,2000)]
        return root,MultirankDataset(root/'ranks'),all_curves

    def test_equal_cutoffs_are_one_case_with_multiple_stage_memberships(self):
        m=self.module();start=1500000000000
        c=Curve(1,10,'normal',start,start+240*H,np.array([start,start+240*H]),np.array([1.,100.]))
        stages=m.origin_groups(c)
        group=next(g for g in stages if g['cutoff']==start+24*H)
        self.assertEqual({s['stage_id'] for s in group['stages']},{'elapsed_24h','progress_10pct'})
        self.assertEqual(len([g for g in stages if g['cutoff']==start+24*H]),1)
        late=next(g for g in stages if g['cutoff']==c.end-12*H)
        self.assertEqual({s['stage_id'] for s in late['stages']},{'remaining_12h','progress_95pct'})

    def test_same_time_grid_grouping_and_blind_prefixes_in_all_splits(self):
        m=self.module();root,base,curves=self.fixture()
        m.build_temporal(root/'temporal',base,curves,source_identity={})
        data=m.TemporalDataset(root/'temporal')
        self.assertEqual(data.manifest['splits']['train']['event_ids'],list(range(100,86,-1)))
        grids=[]
        for split in ('train','validation','test'):
            samples=list(data.inputs(split));cases=data.case_index(split);members=data.memberships(split)
            self.assertEqual(len(samples),len(cases));self.assertFalse(cases.case_id.duplicated().any())
            self.assertEqual(set(cases.case_id),set(members.case_id));grids.append(set(members.stage_id))
            for sample in samples:
                self.assertLessEqual(sample['times'][-1],sample['cutoff']);self.assertNotIn('target',sample)
        self.assertEqual(grids[0],grids[1]);self.assertEqual(grids[1],grids[2])
        coverage=pd.read_csv(data.root/'coverage.csv')
        early=coverage[(coverage.split=='test')&(coverage['rank']==10)&(coverage.stage_id=='elapsed_1h')].iloc[0]
        self.assertEqual(int(early.available_cases),0);self.assertEqual(int(early.missing_prefixes),3)
        with self.assertRaises(ValueError):data.labels('test')
        with self.assertRaises(FileExistsError):m.build_temporal(root/'temporal',base,curves,source_identity={})
        f=data.root/'test_inputs.jsonl.gz';f.write_bytes(f.read_bytes()+b'x')
        with self.assertRaises(ValueError):m.TemporalDataset(data.root)

    def test_missing_or_extra_curves_cannot_silently_change_cohort(self):
        m=self.module();root,base,curves=self.fixture()
        with self.assertRaisesRegex(ValueError,'cohort|missing'):
            m.build_temporal(root/'incomplete',base,curves[:-1],source_identity={})
        self.assertFalse((root/'incomplete').exists())
        extra=replace(curves[0],event=9999)
        with self.assertRaisesRegex(ValueError,'cohort|unexpected'):
            m.build_temporal(root/'extra',base,curves+[extra],source_identity={})

    def test_future_suffix_and_requested_vs_observed_time_are_distinct(self):
        m=self.module();root,base,curves=self.fixture()
        m.build_temporal(root/'one',base,curves,source_identity={})
        changed=[]
        for c in curves:
            scores=c.scores.copy();scores[7:15]+=100000
            changed.append(replace(c,scores=scores))
        m.build_temporal(root/'two',base,changed,source_identity={})
        first=m.TemporalDataset(root/'one');second=m.TemporalDataset(root/'two')
        a=next(s for s in first.inputs('train') if s['curve'].event==100 and s['curve'].rank==10 and s['cutoff']==s['curve'].start+6*H)
        b=next(s for s in second.inputs('train') if s['case_id']==a['case_id'])
        np.testing.assert_array_equal(a['scores'],b['scores'])
        cases=first.case_index('test');last=cases[cases.requested_remaining_h==.25]
        self.assertTrue((last.observation_age_minutes==45).all())
        self.assertTrue((last.observed_remaining_h==1).all())

    def test_frozen_prediction_and_cached_scoring_never_read_answers_or_refit(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_benchmark'),'Temporal benchmark missing')
        m=self.module();root,base,curves=self.fixture()
        from analysis.multirank_model import run_experiment
        from analysis.temporal_benchmark import run_frozen
        from analysis.temporal_evaluate import evaluate
        from unittest.mock import patch
        run_experiment(base,root/'reference',iterations=5)
        m.build_temporal(root/'temporal',base,curves,source_identity={})
        data=m.TemporalDataset(root/'temporal')
        with patch.object(m.TemporalDataset,'labels',side_effect=AssertionError('Answers forbidden')):
            with patch('sqlite3.connect',side_effect=AssertionError('SQLite forbidden')):
                with patch('catboost.CatBoostRegressor.fit',side_effect=AssertionError('Refit forbidden')):
                    run_frozen(data,root/'reference',root/'predictions')
        occupied=root/'occupied';occupied.mkdir();(occupied/'stage_metrics.csv').write_text('preserve')
        with self.assertRaises(FileExistsError):evaluate(root/'predictions',occupied)
        self.assertEqual((occupied/'stage_metrics.csv').read_text(),'preserve')
        scored=evaluate(root/'predictions')
        from analysis.temporal_report import report
        report(root/'predictions',root/'report')
        self.assertTrue((root/'report/manifest.json').exists())
        with self.assertRaises(FileExistsError):report(root/'predictions',root/'report')
        with patch.object(m.TemporalDataset,'labels',side_effect=AssertionError('Do not reopen answers')):
            self.assertEqual(evaluate(root/'predictions'),scored)
        with self.assertRaises(FileExistsError):run_frozen(data,root/'reference',root/'predictions')
        pred=root/'predictions/test_predictions.csv';pred.write_text(pred.read_text()+'\n')
        with self.assertRaises(ValueError):evaluate(root/'predictions')

    def test_trusted_builder_rejects_an_unrelated_archive_before_reading_curves(self):
        m=self.module();root,base,curves=self.fixture()
        from types import SimpleNamespace
        from analysis.holdout_dataset import HoldoutDataset
        archive=root/'other_archive';archive.mkdir();(archive/'download_manifest.json').write_text('{}')
        strict=SimpleNamespace(root=base.root,metadata_directory=base.metadata_directory,
            manifest=base.manifest|{'source_identity':{'download_manifest_sha256':'0'*64}})
        with self.assertRaisesRegex(ValueError,'archive|source'):
            next(m.trusted_curves(strict,HoldoutDataset(root/'base'),archive))

    def test_executed_builder_dependency_sources_are_sealed(self):
        m=self.module();root,base,curves=self.fixture();m.build_temporal(root/'temporal',base,curves,source_identity={})
        data=m.TemporalDataset(root/'temporal');dependency=data.root/'builder_sources/backtest.py'
        self.assertTrue(dependency.exists(),'Executed prefix extraction code must be archived')
        self.assertIn('numpy',data.manifest['build_environment']['software'])
        self.assertEqual(set(data.manifest['protocol']['evaluation_stages']),{s['stage_id'] for s in m.STAGES})
        dependency.write_text(dependency.read_text()+'\n# changed')
        with self.assertRaises(ValueError):m.TemporalDataset(data.root)

    def test_temporal_metrics_deduplicate_and_match_common_event_rank_panels(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.temporal_evaluate'),'Temporal scorer missing')
        from analysis.temporal_evaluate import score_temporal
        cases=pd.DataFrame([
            {'case_id':'a','event':1,'rank':10,'observation_age_minutes':0,'calendar_quarter':'2024Q1'},
            {'case_id':'b','event':1,'rank':10,'observation_age_minutes':0,'calendar_quarter':'2024Q1'},
            {'case_id':'c','event':2,'rank':10,'observation_age_minutes':30,'calendar_quarter':'2024Q2'}])
        members=pd.DataFrame([
            {'case_id':'a','axis':'remaining','stage_id':'remaining_6h'},
            {'case_id':'b','axis':'remaining','stage_id':'remaining_1h'},
            {'case_id':'c','axis':'remaining','stage_id':'remaining_6h'},
            {'case_id':'b','axis':'progress','stage_id':'progress_95pct'}])
        pred=pd.DataFrame([{'case_id':c,'predicted':p,'event':999} for c,p in [('a',120),('b',100),('c',100)]])
        catalog=members[['stage_id','axis']].drop_duplicates().to_dict('records')
        scores=score_temporal(cases,members,{'a':100,'b':100,'c':100},pred,stage_catalog=catalog)
        self.assertAlmostEqual(scores['overall']['event_macro_mape_pct'],5.)
        self.assertEqual(scores['overall']['cases'],3)
        self.assertEqual(scores['common_panels']['remaining']['event_rank_pairs'],1)
        self.assertAlmostEqual(scores['common_panels']['remaining']['by_stage']['remaining_6h']['event_macro_mape_pct'],20.)
        with self.assertRaises(ValueError):score_temporal(cases,members,{'a':100,'b':100,'c':100},pred.iloc[:-1])
        with self.assertRaises(ValueError):score_temporal(cases,members,{'a':100,'b':100,'c':100},pd.concat([pred,pred.iloc[:1]]))

if __name__=='__main__':unittest.main()
