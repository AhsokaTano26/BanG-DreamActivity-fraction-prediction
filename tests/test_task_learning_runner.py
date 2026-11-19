import gzip,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis import multiline_runner,task_peer_data

class TaskLearningRunnerTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.task_learning_runner'),'guarded multi-task runner missing')
        return __import__('analysis.task_learning_runner',fromlist=['run'])

    def test_sealed_peers_graphs_controls_selection_and_resume(self):
        self.module()
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'tests'),str(root)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        done=subprocess.run([sys.executable,'-c','from test_task_learning_runner import TaskLearningRunnerTests; TaskLearningRunnerTests().integration_flow()'],cwd=root,env=env,capture_output=True,text=True,timeout=300)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)

    def test_formal_peer_binding_rejects_resealed_wrong_sources_and_builder(self):
        from types import SimpleNamespace
        from test_task_peer_data import peer_fixture,RANKS
        from analysis.holdout_dataset import sha256
        from analysis.multirank_dataset import identity
        m=self.module();self.assertTrue(hasattr(m,'validate_peer_binding'),'formal peer binding guard missing')
        records,cases=peer_fixture();arrays=task_peer_data.prepare_records(records,cases,RANKS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'data';source.mkdir()
            names=['manifest.json',*[f'{s}_{kind}' for s in ['train','validation','test'] for kind in ['inputs.jsonl.gz','cases.csv']]]
            for name in names:(source/name).write_text('frozen:'+name)
            metadata=SimpleNamespace(root=source,manifest={'dataset_id':'fixture','protocol':{'ranks':RANKS}})
            hashes={name:sha256(source/name) for name in names}
            task_peer_data.save_cache(arrays,root/'valid',source_identity=hashes,dataset_id='fixture');valid=task_peer_data.PeerCache(root/'valid')
            with self.assertRaises(ValueError):m.validate_peer_binding(metadata,valid,allow_fixture=False)
            with patch.object(m,'PEER_CACHE_SEAL',valid.manifest['seal_id']):m.validate_peer_binding(metadata,valid,allow_fixture=False)
            task_peer_data.save_cache(arrays,root/'wrong',source_identity=hashes|{'manifest.json':'0'*64},dataset_id='fixture');wrong=task_peer_data.PeerCache(root/'wrong')
            with patch.object(m,'PEER_CACHE_SEAL',wrong.manifest['seal_id']),self.assertRaises(ValueError):m.validate_peer_binding(metadata,wrong,allow_fixture=False)
            (root/'valid/source.py').write_text('different builder')
            body=json.loads((root/'valid/manifest.json').read_text());body['artifacts']['source.py']=sha256(root/'valid/source.py')
            body['seal_id']=identity({k:v for k,v in body.items() if k!='seal_id'});(root/'valid/manifest.json').write_text(json.dumps(body))
            changed=task_peer_data.PeerCache(root/'valid')
            with patch.object(m,'PEER_CACHE_SEAL',changed.manifest['seal_id']),self.assertRaises(ValueError):m.validate_peer_binding(metadata,changed,allow_fixture=False)

    def integration_flow(self):
        from test_multiline_runner import MultilineRunnerTests
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=MultilineRunnerTests().prepare(root)
            multiline_runner.run(data,root/'reference',base_run=root/'raw',allow_fixture=True)
            records=[]
            for split in ['train','validation','test']:
                with gzip.open(data.root/f'{split}_inputs.jsonl.gz','rt') as f:records.extend(json.loads(line) for line in f)
            cases=pd.concat([data.case_index(s).assign(split=s) for s in ['train','validation','test']],ignore_index=True)
            byid={r['case_id']:r for r in records}
            cases['observed_at']=cases.case_id.map(lambda c:byid[c]['times'][-1]);cases['prefix_points']=cases.case_id.map(lambda c:len(byid[c]['times']))
            task_peer_data.save_cache(task_peer_data.prepare_records(records,cases,data.manifest['protocol']['ranks']),root/'peers',source_identity={'fixture':True},dataset_id=data.manifest['dataset_id'])
            cache=task_peer_data.PeerCache(root/'peers')
            with patch.object(type(data),'labels',side_effect=AssertionError('No independent answer reader')):
                receipt=m.run(data,root/'run',reference_run=root/'reference',peer_cache=cache,allow_fixture=True)
            for split in ['validation','test']:
                a=pd.read_csv(root/f'run/{split}_predictions.csv');b=pd.read_csv(root/f'reference/{split}_predictions.csv')
                self.assertEqual(a.method.nunique(),21)
                for method,p in a.groupby('method'):self.assertEqual(set(p.case_id),set(data.case_index(split).case_id))
                for new,old in m.CONTROLS.items():
                    expected=b[b.method==old].set_index('case_id');observed=a[a.method==new].set_index('case_id').loc[expected.index]
                    np.testing.assert_allclose(observed.predicted,expected.predicted,rtol=1e-12)
            selection=json.loads((root/'run/selection.json').read_text())
            self.assertEqual(set(selection['candidates']),{'ewma_reference','shared_single','rank_heads','related_static','related_phase'})
            for event in [800,3,700,4,600,5]:
                path=root/'run/events'/str(event);audit=json.loads((path/'audit.json').read_text());g=json.loads((path/'relation_state.json').read_text())
                self.assertFalse(audit['current_outcome_released'])
                self.assertTrue(all(r['end_ms']<audit['query_start_ms'] for r in audit['released_forecasts']))
                self.assertNotIn(event,g['released_events'])
                if event in [600,5]:self.assertIsNotNone(audit['selection_sha256'])
                fitted=json.loads((path/'models/config.json').read_text())
                for e,digest in fitted['training_graph_hashes'].items():
                    from analysis.multirank_dataset import identity
                    self.assertEqual(digest,identity(json.loads((root/f'run/events/{e}/relation_state.json').read_text())))
            with patch.object(type(data),'reveal_event',side_effect=AssertionError('Completed resume decodes nothing')):
                self.assertEqual(m.run(data,root/'run',reference_run=root/'reference',peer_cache=cache,allow_fixture=True,resume=True),receipt)
            reveal=data.reveal_event
            def poison(event,*,before_ms):
                curves=reveal(event,before_ms=before_ms)
                if event==800:
                    for c in curves:c.scores[-1]*=2
                return curves
            with patch.object(data,'reveal_event',side_effect=poison):m.run(data,root/'poison',reference_run=root/'reference',peer_cache=cache,allow_fixture=True)
            self.assertEqual((root/'run/events/800/predictions.csv').read_bytes(),(root/'poison/events/800/predictions.csv').read_bytes())

if __name__=='__main__':unittest.main()
