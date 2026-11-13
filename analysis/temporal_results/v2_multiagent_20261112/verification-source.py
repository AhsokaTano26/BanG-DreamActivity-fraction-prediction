import importlib,json
from pathlib import Path
from unittest.mock import patch
import joblib,numpy as np,pandas as pd
from analysis.holdout_dataset import HoldoutDataset,sha256,write_json
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_paired_features import prepare_paired
from analysis.temporal_evaluate import verify_predictions,evaluate
from analysis.research_compare import evaluate as compare
from analysis.research_search import FAMILIES,REFERENCE_METHODS,combine

r=Path('analysis/temporal_results/v2_multiagent_20261112')
old=Path('analysis/temporal_results/v2_paired_tcn_20261112')
seal=verify_predictions(r);verify_predictions(old)
d=TemporalDataset(DEFAULT);v1=HoldoutDataset()
c=json.loads((r/'run_manifest.json').read_text());s=json.loads((r/'selection.json').read_text())
checks={'dataset_id':d.manifest['dataset_id'],'recipes':len(c['recipes']),'total_reported_methods':len(c['methods'])}
for name,digest in c['source_hashes'].items():assert sha256(Path(name))==digest,name
for split in ['train','validation','test']:
 assert d.manifest['splits'][split]['event_ids']==v1.manifest['splits'][split]['event_ids']
 cases=d.case_index(split);pred=pd.read_csv(r/f'{split}_predictions.csv')
 assert set(pred.method)==set(c['methods'])
 for name,g in pred.groupby('method'):
  assert set(g.case_id)==set(cases.case_id) and len(g)==len(cases)
 previous=pd.read_csv(old/f'{split}_predictions.csv')
 for current,prior in REFERENCE_METHODS.items():
  a=pred[pred.method==current].set_index('case_id').predicted.sort_index()
  b=previous[previous.method==prior].set_index('case_id').predicted.sort_index()
  np.testing.assert_allclose(a,b,rtol=1e-14)
 audit=pd.read_csv(r/f'features/{split}_audit.csv');available=audit.peer_available==1
 assert (audit.loc[available,'peer_observed_at']<=audit.loc[available,'observed_at']).all()
 checks[split]={'cases':len(cases),'same_cases_for_all_30_methods':True,'reference_predictions_unchanged':True,'no_future_peer_observation':True}
with patch.object(TemporalDataset,'labels',side_effect=AssertionError('Cached scoring must not read labels')):
 result=evaluate(r);comparison=compare(r)
checks['both_cached_scorers_do_not_read_answers']=True
for row in comparison['comparison']:
 delta=result['splits']['test'][row['method']]['overall']['event_macro_mape_pct']-result['splits']['test'][row['reference']]['overall']['event_macro_mape_pct']
 assert np.isclose(delta,row['new_minus_reference_mape_points'])
test=prepare_paired(d.inputs('test'));train=prepare_paired(d.inputs('train'))
saved=pd.read_csv(r/'test_predictions.csv').pivot(index='case_id',columns='method',values='predicted')
ids=[v['case_id'] for v in test['cases']];predictions={}
for ref in REFERENCE_METHODS:predictions[ref]=saved.loc[ids,ref].to_numpy()
weight_audits=[]
for name,spec in c['recipes'].items():
 module=importlib.import_module(FAMILIES[spec['family']]);bundle=joblib.load(r/f'models/{name}.joblib')
 prediction=module.predict_recipe(bundle,test)
 np.testing.assert_allclose(prediction,saved.loc[ids,name],rtol=1e-10)
 predictions[name]=prediction
 w=bundle['training_weights']
 if isinstance(w,list) and isinstance(w[0],dict):
  table=pd.DataFrame(w);totals=table.groupby('event').weight.sum()
 else:
  table=pd.DataFrame({'event':[v['event'] for v in train['cases']],'weight':w});totals=table.groupby('event').weight.sum()
 assert len(totals)==196 and np.allclose(totals,totals.iloc[0])
 weight_audits.append({'method':name,'events':196,'event_total_weight_equal':True})
 print('Model and event weights verified:',name,flush=True)
for name,spec in s['combination_specs'].items():
 p=combine(spec,predictions,test);np.testing.assert_allclose(p,saved.loc[ids,name],rtol=1e-10)
checks['all_18_models_and_7_combinations_replayed']=True;checks['weights']=weight_audits
assert s['selected_recipe']==min(s['validation'],key=s['validation'].get)
for name,metric in s['validation'].items():assert np.isclose(metric,result['splits']['validation'][name]['overall']['event_macro_mape_pct'])
checks['selected_by_validation_and_independently_reconciled']=True
for receipt in [old/'delivery_manifest.json',r/'evaluation_manifest.json',r/'comparison/evaluation_manifest.json',r/'report/manifest.json']:
 for name,digest in json.loads(receipt.read_text())['artifacts'].items():assert sha256(receipt.parent/name)==digest,name
for name,digest in json.loads(Path('docs/superpowers/plans/2026-11-12-temporal-preservation.json').read_text()).items():assert sha256(Path(name))==digest,name
checks['all_previous_data_database_deliveries_and_new_evaluation_seals_verified']=True
write_json(r/'verification.json',checks)
print('All final verification checks passed',flush=True)
