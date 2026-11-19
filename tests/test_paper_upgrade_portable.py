"""Compact replay trust boundaries, using small real sealed-file fixtures."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from tests.test_paper_upgrade_replay_models import frame,MINIMUM

MODULE='analysis.paper_upgrade_portable'
ROOT=Path(__file__).resolve().parents[1]


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def seal(path,name,fields):
    body=dict(fields);body['artifacts']={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file()};body['seal_id']=identity(body)
    (path/name).write_text(json.dumps(body,sort_keys=True,indent=2)+'\n');return body


def synthetic_export_sources(root):
    from analysis import paper_upgrade_replay_models as m
    fixed=root/'fixed';source=root/'source';fixed.mkdir();source.mkdir()
    cfg={'run_id':'fixture-fixed','dataset_id':'fixture','source_run':str(source),'source_run_id':'fixture-source','methods':list(m.menu()),'warm_forecast_events':[259],'source_hashes':{}}
    for name in ['paper_upgrade_replay_models.py','rolling_residual_stats.py']:
        dest=fixed/'sources/analysis'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'analysis'/name,dest)
        cfg['source_hashes']['analysis/'+name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    history=[];queries=[];parameters=None
    for event,start,split in [(259,MINIMUM,'train'),(294,MINIMUM+1000,'validation'),(308,MINIMUM+2000,'test')]:
        target=frame(event,start);target['split']=split
        past=pd.concat(history,ignore_index=True) if history else target.iloc[:0].assign(log_error=pd.Series(dtype=float),expert__independent=pd.Series(dtype=float))
        detached=past[m.TARGET_COLUMNS+['log_error']].copy()
        detached['center_log_error']=np.log(past.expert__independent/detached.base_prediction)+detached.log_error
        starts=[MINIMUM,MINIMUM+1000,MINIMUM+2000]
        if split=='validation':
            parameters={'complete':m.dlm_parameters(detached)}
            for name,policy in m.POLICIES.items():parameters[name]=m.dlm_parameters(m.feedback_history(detached,query_start_ms=start,policy=policy,query_starts=starts))
            (fixed/'frozen_dlm_parameters.json').write_text(json.dumps(parameters))
        scores,_=m.predict_all(detached,target,query_start_ms=start,parameters=parameters,query_starts=starts)
        fp=fixed/'events'/str(event);sp=source/'events'/str(event);fp.mkdir(parents=True);sp.mkdir(parents=True)
        target.to_csv(fp/'features.csv',index=False)
        # Real first-query empty history predates the added expert columns.
        exported=past.drop(columns=['expert__independent']) if past.empty else past
        exported.to_csv(sp/'released_residuals.csv',index=False)
        refs=pd.concat([pd.DataFrame({'case_id':target.case_id,'method':method,'predicted':values}) for method,values in scores.items()],ignore_index=True)
        refs.to_csv(fp/'predictions.csv',index=False);m.empirical_intervals(detached,target,scores['ewma_reference']).to_csv(fp/'intervals.csv',index=False)
        released=[q|{'prediction_seal_sha256':hashlib.sha256((source/'events'/str(q['event'])/'event_manifest.json').read_bytes()).hexdigest()} for q in queries]
        (sp/'audit.json').write_text(json.dumps({'current_outcome_released':False,'released_forecasts':released}))
        seal(sp,'event_manifest.json',{'run_id':'fixture-source','event':event,'query_start_ms':start,'end_ms':start+100,'split':split})
        audit={'current_outcome_released':False,'released_forecasts':released,'history_cases':len(past),'history_events':past.event.nunique(),
               'source_event_manifest_sha256':hashlib.sha256((sp/'event_manifest.json').read_bytes()).hexdigest(),
               'history_artifact_sha256':hashlib.sha256((sp/'released_residuals.csv').read_bytes()).hexdigest()}
        (fp/'audit.json').write_text(json.dumps(audit))
        seal(fp,'event_manifest.json',{'run_id':'fixture-fixed','event':event,'query_start_ms':start,'end_ms':start+100,'split':split,'cases':len(target),'methods':list(scores)})
        queries.append({'event':event,'start_ms':start,'end_ms':start+100,'split':split})
        released_target=target.assign(log_error=[.1,.2],expert__independent=scores['ewma_reference']);history.append(released_target)
    seal(source,'blind_manifest.json',{'run_id':'fixture-source','dataset_id':'fixture'})
    cfg['source_blind_sha256']=hashlib.sha256((source/'blind_manifest.json').read_bytes()).hexdigest()
    (fixed/'run_manifest.json').write_text(json.dumps(cfg))
    seal(fixed,'blind_manifest.json',{'run_id':'fixture-fixed','dataset_id':'fixture','methods':list(m.menu())})
    return fixed,source


class PortableReplay(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE),'portable trusted exporter/replayer must exist')
        self.m=importlib.import_module(MODULE)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_future_error_cannot_affect_earlier_prediction(self):
        from analysis import paper_upgrade_replay_models as models
        history=pd.concat([frame(1,MINIMUM,{1000:.1,2000:.2}),frame(2,MINIMUM+1000,{1000:.3,2000:.4})],ignore_index=True)
        history['expert__independent']=195.
        target=frame(2,MINIMUM+1000);query={'event':2,'start_ms':MINIMUM+1000,'history_cases':2,'history_case_ids_sha256':self.m.case_ids_hash(['1:1000','1:2000'])}
        early=self.m.history_for_query(history,query)
        altered=history.copy();altered.loc[altered.event==2,'log_error']=1000.
        after=self.m.history_for_query(altered,query)
        first,_=models.predict_all(early,target,query_start_ms=MINIMUM+1000)
        second,_=models.predict_all(after,target,query_start_ms=MINIMUM+1000)
        for method in first:np.testing.assert_array_equal(first[method],second[method])

    def test_duplicate_or_missing_case_history_rejected(self):
        h=frame(1,MINIMUM,{1000:.1,2000:.2}).assign(expert__independent=200)
        query={'event':2,'start_ms':MINIMUM+1000,'history_cases':2,'history_case_ids_sha256':self.m.case_ids_hash(h.case_id)}
        with self.assertRaises(ValueError):self.m.history_for_query(pd.concat([h,h]),query)
        with self.assertRaises(ValueError):self.m.history_for_query(h.iloc[:1],query)

    def test_same_case_conflicting_feedback_across_snapshots_rejected(self):
        h=frame(1,MINIMUM,{1000:.1,2000:.2}).assign(expert__independent=200)
        changed=h.copy();changed.loc[0,'log_error']+=.01
        with self.assertRaises(ValueError):self.m.merge_history(h,changed)

    def test_source_seal_detects_tampered_artifact(self):
        source=self.root/'sealed';source.mkdir();(source/'input.csv').write_text('x\n1\n');seal(source,'manifest.json',{})
        (source/'input.csv').write_text('x\n2\n')
        with self.assertRaises(ValueError):self.m.verify_seal(source/'manifest.json')

    def test_fixture_capsule_replays_without_original_workspace(self):
        fixed,_=synthetic_export_sources(self.root);capsule=self.root/'capsule'
        self.m.export_capsule(fixed,capsule,allow_fixture=True)
        manifest=json.loads((capsule/'capsule_manifest.json').read_text())
        self.assertFalse(any(name.endswith('.pyc') for name in manifest['artifacts']))
        shutil.rmtree(fixed);shutil.rmtree(self.root/'source')
        output=self.root/'replayed'
        result=subprocess.run([sys.executable,'-I','-c',"import sys;sys.path.insert(0,sys.argv[1]);from analysis.paper_upgrade_portable import replay_capsule;print(replay_capsule(sys.argv[1],sys.argv[2])['status'])",str(capsule),str(output)],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        receipt=json.loads((output/'replay_receipt.json').read_text())
        self.assertEqual(receipt['status'],'pass');self.assertEqual(receipt['events'],3);self.assertEqual(receipt['methods'],54)
        self.assertTrue((capsule/'analysis/rolling_residual_stats.py').exists())

if __name__=='__main__':unittest.main()
