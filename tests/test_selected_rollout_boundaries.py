import importlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.backtest import ROOT,Curve
from analysis.data_policy import MIN_EVENT_START_MS as S
from analysis.holdout_dataset import sha256
from analysis.rolling_benchmark import seal_directory
from production.state import Ledger
H=3600000

def loaded_text(plist,args,*,directory=ROOT,program=None):
    return (f'gui/{os.getuid()}/com.tano.bangdream-forecast = {{\n'
        f'\tpath = {plist}\n\tprogram = {program or args[0]}\n\targuments = {{\n'+
        ''.join('\t\t'+v+'\n' for v in args)+f'\t}}\n\tworking directory = {directory}\n}}\n')

def seed_fixture(tmp):
    from production import selected_feedback as m
    run=Path(tmp)/'sealed';curves=[]
    def event(e,start,raw):
        directory=run/'events'/str(e);directory.mkdir(parents=True)
        pd.DataFrame([dict(case_id=f'{e}:1000',event=e,rank=1000,phase=1,start_ms=start,end_ms=start+H,
            current=50.,base_prediction=raw)]).to_csv(directory/'features.csv',index=False)
        seal_directory(directory,'event_manifest.json',dict(run_id=m.REFERENCE_ID,event=e,query_start_ms=start,end_ms=start+H))
        curves.append(Curve(e,1000,'normal',start,start+H,np.array([start+H//2,start+H]),np.array([50.,100.])))
    event(900,S,110.)
    seal_directory(run,'blind_manifest.json',dict(run_id=m.REFERENCE_ID,protocol='rolling_multiline_calibration_v3'))
    digest=sha256(run/'blind_manifest.json')
    store=SimpleNamespace(load_curves=lambda *a,**k:curves)
    root=Path(tmp)/'live';ledger=Ledger(root)
    cfg=dict(root=str(root),ranks=[1000],selected_seed_run=str(run))
    return m,run,digest,ledger,cfg,store,event

class SelectedRolloutBoundaryTests(unittest.TestCase):
    def test_loaded_identity_requires_our_path_arguments_program_and_directory(self):
        from production import selected_schedule as m
        self.assertTrue(callable(getattr(m,'validate_loaded',None)),'loaded-job identity guard missing')
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(root=tmp,interval_seconds=900);path=(Path(tmp)/'worker.plist').resolve();config=(Path(tmp)/'config.json').resolve()
            old=[str(ROOT/'.venv/bin/python'),'-m','production.cli','cycle','--config',str(config)]
            new=['/usr/bin/nice','-n','19',str(ROOT/'.venv/bin/python'),'-m','production.selected_cli','cycle','--config',str(config)]
            for args in [old,new]:
                identity=m.validate_loaded(loaded_text(path,args),cfg,config,path)
                self.assertEqual(identity['arguments'],args)
            bad=[loaded_text(path,old,directory='/foreign'),loaded_text(path,old,program='/foreign/python'),
                 loaded_text(Path(tmp)/'other.plist',old),loaded_text(path,old+['foreign']),
                 loaded_text(path,old).replace(f'\tworking directory = {ROOT}\n','')]
            for text in bad:
                with self.subTest(text=text),self.assertRaises(FileExistsError):m.validate_loaded(text,cfg,config,path)

    def test_install_and_uninstall_refuse_loaded_foreign_job_with_missing_plist(self):
        from production import selected_schedule as m
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)/'home';root=Path(tmp)/'live';(root/'models').mkdir(parents=True)
            config_path=root/'config.json';cfg=dict(root=str(root),interval_seconds=900,forecast_method='legacy_pair')
            config_path.write_text(json.dumps(cfg));(root/'models/current.json').write_text(json.dumps({'path':str(root/'pack')}))
            path=home/'Library/LaunchAgents/com.tano.bangdream-forecast.plist'
            (root/'scheduler.json').write_text(json.dumps(dict(plist=str(path),config=str(config_path),domain=f'gui/{os.getuid()}')))
            text=loaded_text(path,['/foreign/python','foreign.py'],directory='/foreign')
            def command(args,**kwargs):
                if args[:2]==['launchctl','print']:return SimpleNamespace(returncode=0,stdout=text,stderr='')
                raise AssertionError('Foreign loaded service must never be stopped or changed')
            with patch.object(m.Path,'home',return_value=home),patch.object(m,'verify_pack',return_value={'production_ready':True}),patch.object(m.old,'_command',side_effect=command):
                for action in [lambda:m.install(cfg,config_path),lambda:m.uninstall(cfg)]:
                    with self.subTest(action=action),self.assertRaises(FileExistsError):action()
            self.assertFalse(path.exists())

    def test_uninstall_does_not_interrupt_a_running_prediction_cycle(self):
        from production import selected_schedule as m
        from production.runtime import cycle_lock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json'
            (root/'scheduler.json').write_text(json.dumps(dict(plist=str(root/'job.plist'),config=str(config),domain=f'gui/{os.getuid()}')))
            with cycle_lock(root) as acquired:
                self.assertTrue(acquired)
                with patch.object(m.old,'_command',side_effect=AssertionError('Must not stop a busy cycle')):
                    with self.assertRaises(RuntimeError):m.uninstall(dict(root=str(root),interval_seconds=900))

    def test_seed_ignores_unlisted_event_package_even_with_same_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,run,digest,ledger,cfg,store,event=seed_fixture(tmp)
            event(9,S+2*H,9999.)  # valid own seal, but not a frozen outer member
            with patch.object(m,'REFERENCE_MANIFEST_SHA256',digest,create=True):
                m.seed_feedback(store,ledger,cfg,before_ms=S+5*H,now_ms=S+6*H)
            self.assertEqual([r['event'] for r in ledger.feedback_rows(before_ms=S+5*H)],[900])

    def test_completed_seed_restores_receipt_after_interruption_without_rereading_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,run,digest,ledger,cfg,store,event=seed_fixture(tmp)
            with patch.object(m,'REFERENCE_MANIFEST_SHA256',digest,create=True):
                m.seed_feedback(store,ledger,cfg,before_ms=S+5*H,now_ms=S+6*H)
                receipt=Path(cfg['root'])/'selected_seed.json';expected=json.loads(receipt.read_text());receipt.unlink()
                run.rename(Path(tmp)/'source-archived')
                again=m.seed_feedback(store,ledger,cfg,before_ms=S+5*H,now_ms=S+6*H)
            self.assertEqual(again['status'],'not_needed')
            self.assertTrue(receipt.exists(),'completed DB receipt must restore interrupted file publication')
            self.assertEqual(json.loads(receipt.read_text()),expected)

if __name__=='__main__':unittest.main()
