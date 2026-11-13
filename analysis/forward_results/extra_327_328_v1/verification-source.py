import json
from pathlib import Path
from unittest.mock import patch
import joblib,numpy as np,pandas as pd
from analysis.holdout_dataset import HoldoutDataset,sha256,write_json
from analysis.temporal_dataset import TemporalDataset
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_evaluate import verify_predictions
from analysis.forward_validation import ForwardDataset,verify_archive
from analysis.forward_evaluate import evaluate
from analysis.temporal_paired_features import prepare_paired
from analysis.multirank_model import model_prediction
from analysis.research_search import FAMILIES,combine
import importlib

r=Path('analysis/forward_results/extra_327_328_v1');data=ForwardDataset('data/forward_validation/extra_327_328_v1')
v1=HoldoutDataset();parent=TemporalDataset(DEFAULT);seal=verify_predictions(r)
config=json.loads((r/'run_manifest.json').read_text());checks={}
assert data.manifest['parent_dataset_id']==parent.manifest['dataset_id']
blocked={e for s in v1.manifest['splits'].values() for e in s['event_ids']}
assert not blocked.intersection(data.manifest['events'])
for n,digest in config['source_hashes'].items():assert sha256(Path(n))==digest,n
for role,item in config['model_identities'].items():
 p=Path(item['directory']);verify_predictions(p);assert sha256(p/'blind_manifest.json')==item['blind_manifest_sha256']
checks['all_model_identities_sources_and_parent_dataset_verified']=True
source=Path(data.manifest['archive']);verify_archive(source)
metadata=json.loads((source/'metadata.json').read_text());actual=data.outcomes(allow=True)
for row in metadata['rows']:
 info=row['event_info'];payload=json.loads((source/row['raw_file']).read_text());points={}
 for p in payload['cutoffs']:
  t,ep=int(p['time']),int(p['ep'])
  assert t not in points or points[t]==ep
  points[t]=ep
 times=[t for t in points if info['end_ms']-60000<=t<=info['end_ms']+300000]
 saved=actual[(actual.event==row['event'])&(actual['rank']==row['rank'])].iloc[0]
 if times:assert saved.actual==points[max(times)] and saved.observed_at==max(times)
 else:assert pd.isna(saved.actual) and saved.status=='missing_near_end'
checks['finals_independently_recomputed_from_primary_raw_responses']=True
with patch.object(ForwardDataset,'outcomes',side_effect=AssertionError('Cached evaluation reopened answers')):
 result=evaluate(r)
assert result['scored_events']==[328] and result['pending_events']==[327]
assert result['full_cohort_event_mape_pct'] is None
checks['pending_event_never_scored_and_cached_evaluation_does_not_reopen_outcomes']=True
parent_config=json.loads((Path(config['model_run'])/'run_manifest.json').read_text())
modules={f:importlib.import_module(path) for f,path in FAMILIES.items()}
for group in ['grid','legacy']:
 arrays=prepare_paired(data.inputs(group));cases=data.cases(group)
 pred=pd.read_csv(r/f'{group}_predictions.csv');ids=[c['case_id'] for c in arrays['cases']]
 assert len(cases)==len(ids) and set(pred.case_id)==set(ids)
 for m,p in pred.groupby('method'):assert len(p)==len(ids) and not p.case_id.duplicated().any()
 lookup=pred.pivot(index='case_id',columns='method',values='predicted')
 parts={}
 for name in config['combination']['members']:
  model=joblib.load(r/f'models/{name}.joblib');family=parent_config['recipes'][name]['family']
  parts[name]=modules[family].predict_recipe(model,arrays)
 expected={'selected_combo':combine(config['combination'],parts,arrays),
  'reference_single':model_prediction(joblib.load(r/'models/reference_single.joblib'),{'rank':arrays['single'],'current':arrays['current'],'progress':arrays['progress']}),
  'reference_paired':model_prediction(joblib.load(r/'models/reference_paired.joblib'),arrays)}
 for method,values in expected.items():np.testing.assert_allclose(values,lookup.loc[ids,method],rtol=1e-10)
 audit=pd.read_csv(r/f'{group}_pair_audit.csv');valid=audit.peer_available==1
 assert (audit.loc[valid,'peer_observed_at']<=audit.loc[valid,'observed_at']).all()
 checks[group]={'prefixes':len(ids),'all_three_methods_same_cases':True,'saved_model_replay_exact':True,'no_future_peer_observations':True}
for receipt in [r/'evaluation_manifest.json',r/'report_manifest.json',Path('analysis/temporal_results/v2_multiagent_20261112/delivery_manifest.json')]:
 for n,digest in json.loads(receipt.read_text())['artifacts'].items():assert sha256(receipt.parent/n)==digest,n
for n,digest in json.loads(Path('docs/superpowers/plans/2026-11-12-temporal-preservation.json').read_text()).items():assert sha256(Path(n))==digest,n
checks['old_data_database_and_deliveries_unchanged']=True
write_json(r/'verification.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
