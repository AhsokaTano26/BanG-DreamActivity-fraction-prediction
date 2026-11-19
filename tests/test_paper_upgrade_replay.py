"""A real sealed-file fixture exercises the trusted replay gateway."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from tests.test_paper_upgrade_replay_models import frame,MINIMUM
from analysis.holdout_dataset import sha256
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import seal_directory,write_new_json

MODULE='analysis.paper_upgrade_replay'


def fixture(root):
    base=root/'base';source=root/'source';data=root/'data'
    base.mkdir();source.mkdir();data.mkdir()
    rows=[];targets={}
    for event,start,split in [(1,MINIMUM,'train'),(2,MINIMUM+1000,'validation')]:
        t=frame(event,start);t['split']=split;targets[event]=t
        rows.append(dict(event=event,start_ms=start,end_ms=start+100,split=split,sha256='unused'))
    write_new_json(data/'manifest.json',{'dataset_id':'fixture','history_index':rows,'splits':{
        'train':{'event_ids':[1]},'validation':{'event_ids':[2]},'test':{'event_ids':[]}}})
    for split in ['train','validation','test']:
        parts=[t for e,t in targets.items() if t.split.iloc[0]==split]
        pd.concat(parts,ignore_index=True).to_csv(data/f'{split}_cases.csv',index=False) if parts else pd.DataFrame(columns=targets[1].columns).to_csv(data/f'{split}_cases.csv',index=False)
    for e in [1,2]:
        bp=base/'events'/str(e);sp=source/'events'/str(e);bp.mkdir(parents=True);sp.mkdir(parents=True)
        t=targets[e];r=rows[e-1]
        t.assign(predicted=t.base_prediction).to_csv(bp/'predictions.csv',index=False)
        write_new_json(bp/'audit.json',{'current_outcome_released':False,'released_history':rows[:e-1]})
        seal_directory(bp,'event_manifest.json',{'run_id':'baseid','event':e,'query_start_ms':r['start_ms'],'end_ms':r['end_ms'],'split':r['split']})
        t.to_csv(sp/'features.csv',index=False)
        pd.concat([t.assign(method=m,predicted=190. if e==1 and m=='ewma_reference' else t.base_prediction) for m in ['uncorrected','ewma_reference','legacy_pair']]).to_csv(sp/'predictions.csv',index=False)
        h=targets[1].assign(log_error=.1,expert__independent=190.) if e==2 else targets[1].iloc[:0].assign(log_error=pd.Series(dtype=float),expert__independent=pd.Series(dtype=float))
        h.to_csv(sp/'released_residuals.csv',index=False)
        released=[r|{'prediction_seal_sha256':sha256(source/'events'/str(r['event'])/'event_manifest.json')} for r in rows[:e-1]]
        write_new_json(sp/'audit.json',{'event':e,'query_start_ms':r['start_ms'],'current_outcome_released':False,'released_forecasts':released,'base_event_manifest_sha256':sha256(bp/'event_manifest.json')})
        seal_directory(sp,'event_manifest.json',{'run_id':'sourceid','event':e,'query_start_ms':r['start_ms'],'end_ms':r['end_ms'],'split':r['split']})
    write_new_json(base/'run_manifest.json',{'run_id':'baseid','dataset_id':'fixture'})
    seal_directory(base,'blind_manifest.json',{'run_id':'baseid','dataset_id':'fixture'})
    write_new_json(source/'run_manifest.json',{'run_id':'sourceid','dataset_id':'fixture','base_run':str(base),'base_run_id':'baseid','warm_forecast_events':[1],'warmup_events':[],'exposure':'fixture','metadata_caveat':'fixture'})
    seal_directory(source,'blind_manifest.json',{'run_id':'sourceid','dataset_id':'fixture'})
    return base,source,data


class ReplayGateway(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE),'trusted replay gateway must exist')
        self.m=importlib.import_module(MODULE)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.base,self.source,self.data=fixture(self.root)

    def test_gateway_preserves_routed_base_and_released_clock(self):
        g=self.m.SourceGateway(self.source,self.base,self.data,allow_fixture=True)
        record=g.event_record(2)
        self.assertEqual(set(record['history'].event),{1})
        self.assertTrue((record['history'].end_ms<MINIMUM+1000).all())
        self.assertEqual(record['target'].base_prediction.tolist(),[200.,200.])
        self.assertNotIn('expert__independent',record['target'])
        self.assertAlmostEqual(record['history'].center_log_error.iloc[0],.1+__import__('math').log(190/200))

    def test_gateway_rejects_tampered_artifact(self):
        g=self.m.SourceGateway(self.source,self.base,self.data,allow_fixture=True)
        p=self.source/'events/2/features.csv';p.write_text(p.read_text().replace('200.0','201.0'))
        with self.assertRaises(ValueError):g.event_record(2)

    def test_explicit_base_parity_rejects_case_routing_change(self):
        a=frame(2,MINIMUM+1000);b=a.copy();b.loc[0,'base_prediction']=201
        with self.assertRaises(ValueError):self.m.case_parity(a,b)

    def test_clock_guard_rejects_end_equal_query_even_with_claimed_release(self):
        rows=[{'event':1,'start_ms':MINIMUM,'end_ms':MINIMUM+1000}]
        with self.assertRaises(ValueError):self.m.release_guard(rows,MINIMUM+1000,2)

    def test_numeric_context_stays_identical_across_csv_inference_scopes(self):
        first=frame(1,MINIMUM,{1000:.1,2000:.1});first['band_group']='2'
        second=first.copy();second['case_id']=second.case_id+'b'
        other=frame(4,MINIMUM+400,{1000:.1,2000:.1});other['band_group']='2:3'
        a=self.root/'only_numeric.csv';b=self.root/'mixed_context.csv'
        first.to_csv(a,index=False);pd.concat([second,other]).to_csv(b,index=False)
        history=pd.concat([self.m.read(a),self.m.read(b)],ignore_index=True)
        self.m.models.validate(history,frame(3,MINIMUM+1000),MINIMUM+1000)
        self.assertEqual(history[history.event==1].band_group.nunique(),1)

if __name__=='__main__':unittest.main()
