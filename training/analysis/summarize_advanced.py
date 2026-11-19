"""Validate and summarize the completed advanced/transfer experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'analysis/advanced_results'
NEURAL = ['mlp','gru','lstm','tcn','transformer','patchtst','tsmixer','itransformer','dlinear']
LABELS = {
    'linear':'平均速度外推', 'shape_analog':'历史相似曲线', 'xgb_tab_growth':'XGBoost·表格特征',
    'xgb_sequence_residual':'XGBoost·序列残差', 'catboost_sequence_residual':'CatBoost·序列残差',
    'lightgbm_sequence_residual':'LightGBM·序列残差', 'extra_trees_sequence_residual':'ExtraTrees·序列残差',
    'validation_top3_ensemble':'GRU＋TCN＋DLinear组合',
    'pretrain_finetune_tcn_ensemble3':'TCN·旧数据预训练后微调',
    'weighted_history_tcn_ensemble3':'TCN·旧数据降权联合训练',
}
for name in NEURAL:
    LABELS[name+'_ensemble3'] = {'mlp':'MLP','gru':'GRU','lstm':'LSTM','tcn':'TCN',
        'transformer':'Transformer','patchtst':'PatchTST','tsmixer':'TSMixer',
        'itransformer':'iTransformer','dlinear':'DLinear混合适配'}[name]+'·三种子集成'


def main():
    main_report=json.load(open(OUT/'benchmark.json'))
    transfer_dir=ROOT/'analysis/transfer_results'
    transfer_report=json.load(open(transfer_dir/'transfer.json'))
    main_data=pd.read_csv(OUT/'predictions.csv');main_data=main_data[main_data.phase=='test']
    transfer=pd.read_csv(transfer_dir/'predictions.csv')
    fingerprint=hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest()
    assert fingerprint==main_report['database_sha256']==transfer_report['database_sha256']
    assert len(main_report['training_runs'])==142 and len(transfer_report['training_runs'])==24
    for report in [main_report,transfer_report]:
        for audit in report['fold_audit']:
            end=audit.get('training_latest_end',audit.get('latest_training_end'))
            start=audit.get('test_earliest_start',audit.get('earliest_test_start'))
            assert end+15*60000<start
    all_data=pd.concat([main_data,transfer],ignore_index=True)
    assert (all_data.groupby(['method','seed']).size()==342).all()
    assert not all_data.duplicated(['method','seed','event','rank','stage']).any()
    assert np.isfinite(all_data.predicted).all() and (all_data.predicted>=all_data.current-1e-6).all()
    assert (all_data.observed_at<=all_data.cutoff).all()
    reference=main_data[main_data.method=='linear'].set_index(['event','rank','stage'])
    for (_, _),group in all_data.groupby(['method','seed']):
        common=group.set_index(['event','rank','stage']).sort_index()
        expected=reference.sort_index()
        for column in ['actual','current','cutoff','observed_at']:
            np.testing.assert_array_equal(common[column].to_numpy(),expected[column].to_numpy())
    all_data['ape']=abs(all_data.predicted/all_data.actual-1)*100
    summaries=[]
    for method,label in LABELS.items():
        frame=all_data[all_data.method==method]
        assert len(frame)==342
        base=method.rsplit('_ensemble3',1)[0]
        repeated=all_data[(all_data.method==base)&all_data.seed.isin([42,123,2026])]
        individual=repeated.groupby('seed').ape.mean()
        summaries.append({'method':method,'label':label,'n':len(frame),
            'mape':float(frame.ape.mean()),'median_ape':float(frame.ape.median()),
            'p90_ape':float(frame.ape.quantile(.9)),
            'rank1000_mape':float(frame[frame['rank']==1000].ape.mean()),
            'rank2000_mape':float(frame[frame['rank']==2000].ape.mean()),
            'individual_seed_mape_mean':float(individual.mean()) if len(individual)==3 else None,
            'individual_seed_mape_std':float(individual.std()) if len(individual)==3 else None})
    summary=pd.DataFrame(summaries).sort_values('mape')
    summary.to_csv(OUT/'comparison-summary.csv',index=False)
    combined={'database_sha256':fingerprint,'training_runs':166,'main_training_runs':142,
        'transfer_training_runs':24,'test_events':36,'test_snapshots':342,
        'older_transfer_events':transfer_report['old_events'],
        'older_transfer_curves':transfer_report['old_curves'],
        'total_measured_training_seconds':sum(r['seconds'] for r in main_report['training_runs']+transfer_report['training_runs']),
        'comparison_type':'exploratory; four chronological expanding-window blocks; each event excluded from its own training',
        'summary':summary.where(pd.notna(summary),None).to_dict(orient='records')}
    # Convert optional pandas float NaNs into JSON nulls.
    for row in combined['summary']:
        for key,value in row.items():
            if isinstance(value,float) and not np.isfinite(value):row[key]=None
    (OUT/'comparison-summary.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2,allow_nan=False))
    print(summary[['label','mape','rank1000_mape','rank2000_mape','individual_seed_mape_std']].round(4).to_string(index=False))
    print('VALIDATED',json.dumps({k:v for k,v in combined.items() if k!='summary'},ensure_ascii=False))


if __name__=='__main__':
    main()
