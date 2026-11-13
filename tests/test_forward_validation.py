import gzip,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from datetime import datetime,timezone
from analysis.holdout_dataset import sha256,write_json

H=3600000

class ForwardValidationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.forward_validation'))
        from analysis import forward_validation
        return forward_validation

    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        base=root/'base';base.mkdir();pd.DataFrame([{'end_ms':1000}]).to_csv(base/'partitions.csv',index=False)
        parent=SimpleNamespace(root=base,manifest={'dataset_id':'parent','splits':{'train':{'event_ids':[1]},'validation':{'event_ids':[2]},'test':{'event_ids':[3]}}})
        archive=root/'archive';archive.mkdir();rows=[]
        for event in [4,5]:
            for rank in [1000,2000]:
                start=1000000+event*300*H;end=start+200*H
                times=[start+i*H for i in range(201 if event==5 else 200)]
                points=[{'time':t,'ep':(i+1)*100} for i,t in enumerate(times)]
                filename=f'{event}_{rank}.json';write_json(archive/filename,{'result':True,'cutoffs':points})
                rows.append({'event':event,'rank':rank,'raw_file':filename,'raw_sha256':sha256(archive/filename),
                             'event_info':{'start_ms':start,'end_ms':end,'event_type':'normal','name':str(event)}})
        write_json(archive/'metadata.json',{'rows':rows,'ongoing_activities':[],'unstarted_announced_activities':[]})
        write_json(archive/'manifest.json',{'artifacts':{p.name:sha256(p) for p in archive.iterdir()}})
        return root,parent,archive

    def test_new_events_freeze_all_origins_and_missing_outcomes_without_leakage(self):
        m=self.module();root,parent,archive=self.fixture()
        m.build(archive,root/'data',parent)
        data=m.ForwardDataset(root/'data');cases=data.cases('grid')
        self.assertEqual(set(cases.event),{4,5});self.assertEqual(len(data.memberships().stage_id.unique()),20)
        self.assertEqual(set(data.outcomes(allow=True).query("status == 'available_near_end'").event),{5})
        with self.assertRaises(ValueError):data.outcomes()
        for s in data.inputs('grid'):
            self.assertLessEqual(s['times'][-1],s['cutoff']);self.assertNotIn('actual',s)
        with self.assertRaises(FileExistsError):m.build(archive,data.root,parent)
        f=data.root/'grid_inputs.jsonl.gz';f.write_bytes(f.read_bytes()+b'x')
        with self.assertRaises(ValueError):m.ForwardDataset(data.root)

    def test_builder_rejects_cohort_overlap_before_writing_output(self):
        m=self.module();root,parent,archive=self.fixture()
        parent.manifest['splits']['test']['event_ids']=[4]
        with self.assertRaises(ValueError):m.build(archive,root/'invalid',parent)
        self.assertFalse((root/'invalid').exists())

    def test_changing_only_final_answer_cannot_change_blind_prefixes(self):
        import shutil
        m=self.module();root,parent,archive=self.fixture();m.build(archive,root/'first',parent)
        newer=root/'newer';shutil.copytree(archive,newer)
        metadata=json.loads((newer/'metadata.json').read_text());row=metadata['rows'][2]
        p=newer/row['raw_file'];raw=json.loads(p.read_text());raw['cutoffs'][-1]['ep']=40200;write_json(p,raw)
        row['raw_sha256']=sha256(p);write_json(newer/'metadata.json',metadata)
        write_json(newer/'manifest.json',{'artifacts':{p.name:sha256(p) for p in newer.iterdir() if p.name!='manifest.json'}})
        m.build(newer,root/'second',parent)
        a=gzip.decompress((root/'first/grid_inputs.jsonl.gz').read_bytes())
        b=gzip.decompress((root/'second/grid_inputs.jsonl.gz').read_bytes())
        self.assertEqual(a,b)
        self.assertNotEqual((root/'first/outcomes.csv').read_bytes(),(root/'second/outcomes.csv').read_bytes())

    def test_historical_zero_is_allowed_and_builder_dependencies_are_sealed(self):
        m=self.module();root,parent,archive=self.fixture()
        row=json.loads((archive/'metadata.json').read_text());item=row['rows'][2]
        p=archive/item['raw_file'];raw=json.loads(p.read_text());raw['cutoffs'][0]['ep']=0;write_json(p,raw)
        item['raw_sha256']=sha256(p);write_json(archive/'metadata.json',row)
        write_json(archive/'manifest.json',{'artifacts':{p.name:sha256(p) for p in archive.iterdir() if p.name!='manifest.json'}})
        m.build(archive,root/'data',parent);data=m.ForwardDataset(root/'data')
        source=data.root/'builder_sources/download_ranks.py'
        self.assertTrue(source.exists())
        source.write_text(source.read_text()+'\n')
        with self.assertRaises(ValueError):m.ForwardDataset(data.root)

    def test_prediction_rejects_a_model_from_another_parent_dataset(self):
        import platform
        m=self.module();root,parent,archive=self.fixture();m.build(archive,root/'data',parent)
        dirs=[]
        for i in range(3):
            run=root/f'model{i}';run.mkdir()
            write_json(run/'run_manifest.json',{'dataset_id':'wrong','source_hashes':{},'software':{},'python':platform.python_version()})
            write_json(run/'blind_manifest.json',{'dataset_id':'wrong','artifacts':{'run_manifest.json':sha256(run/'run_manifest.json')}})
            dirs.append(run)
        with self.assertRaisesRegex(ValueError,'parent|dataset'):
            m.predict(m.ForwardDataset(root/'data'),*dirs,root/'predictions')
        self.assertFalse((root/'predictions').exists())

    def test_legacy_snapshot_uses_its_sealed_event_schedule(self):
        m=self.module();root,parent,archive=self.fixture();legacy=root/'legacy';legacy.mkdir()
        rows=json.loads((archive/'metadata.json').read_text())['rows'];info=rows[2]['event_info']
        cutoff=info['start_ms']+194*H;old_start=info['start_ms']-H
        iso=lambda t:datetime.fromtimestamp(t/1000,timezone.utc).isoformat()
        predictions=[]
        for rank in [1000,2000]:
            row=next(r for r in rows if r['event']==5 and r['rank']==rank)
            raw=json.loads((archive/row['raw_file']).read_text());raw['cutoffs']=raw['cutoffs'][:195]
            write_json(legacy/f'rank_{rank}.json',raw)
            predictions.append({'rank':rank,'observed_at':iso(cutoff),'current_ep':19500,'predicted_final_ep':20100})
        write_json(legacy/'prediction.json',{'event_id':5,'as_of':iso(cutoff),'event_start':iso(old_start),
            'event_end':iso(info['end_ms']),'event_type':'live_try','predictions':predictions})
        write_json(legacy/'prediction_manifest.json',{'artifacts':{p.name:sha256(p) for p in legacy.iterdir()}})
        m.build(archive,root/'data',parent,legacy)
        data=m.ForwardDataset(root/'data')
        self.assertTrue((data.cases('legacy').start_ms==old_start).all())
        self.assertTrue(all(s['curve'].kind=='live_try' for s in data.inputs('legacy')))

    def test_scoring_reports_incomplete_cohort_and_idempotent_labels(self):
        m=self.module();root,parent,archive=self.fixture();m.build(archive,root/'data',parent)
        self.assertIsNotNone(importlib.util.find_spec('analysis.forward_evaluate'))
        from analysis.forward_evaluate import evaluate
        from unittest.mock import patch
        data=m.ForwardDataset(root/'data');run=root/'run';run.mkdir()
        cases=data.cases('grid');pred=[]
        for method in ['reference_single','reference_paired','selected_combo']:
            pred.extend({'case_id':r.case_id,'method':method,'predicted':20100.} for r in cases.itertuples())
        pd.DataFrame(pred).to_csv(run/'grid_predictions.csv',index=False)
        write_json(run/'blind_manifest.json',{'dataset_directory':str(data.root),'dataset_id':data.manifest['dataset_id'],
            'methods':['reference_single','reference_paired','selected_combo'],
            'artifacts':{'grid_predictions.csv':sha256(run/'grid_predictions.csv')}})
        result=evaluate(run)
        self.assertIsNone(result['full_cohort_event_mape_pct'])
        self.assertEqual(result['scored_events'],[5]);self.assertEqual(result['pending_events'],[4])
        self.assertAlmostEqual(result['grid']['selected_combo']['overall']['event_macro_mape_pct'],0.)
        with patch.object(m.ForwardDataset,'outcomes',side_effect=AssertionError('Cached labels reopened')):
            self.assertEqual(evaluate(run),result)
