"""Validate context experiments and export a comparison with attribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/context_results'
LABELS={
 'lgbm_baseline':'原LightGBM基线','lgbm_band':'LightGBM＋乐队标签',
 'lgbm_static':'LightGBM＋乐队/角色/卡牌资料','lgbm_history':'LightGBM＋分组历史统计',
 'lgbm_peer':'LightGBM＋历史统计/跨档走势','catboost_peer':'CatBoost＋完整上下文（验证集选定）',
 'catboost_history':'CatBoost＋上下文历史','catboost_peer_regularized':'CatBoost＋更强正则',
 'lgbm_type_experts':'按活动类型分专家','tcn_context_ensemble3':'完整上下文TCN集成',
 'cb_baseline':'CatBoost基线','cb_global_history':'CatBoost＋整体历史',
 'cb_type_history':'CatBoost＋整体/类型历史','cb_band_history':'CatBoost＋类型/乐队历史',
 'cb_character_history':'CatBoost＋类型/乐队/角色历史','cb_compact_peer':'精简分组历史＋跨档走势',
 'tcn_compact_ensemble3':'精简上下文TCN集成','tcn_conditional_prior_ensemble3':'历史分组先验TCN集成'}


def main():
    first=json.load(open(OUT/'benchmark.json'))
    second_dir=ROOT/'analysis/context_ablation_results';second=json.load(open(second_dir/'benchmark.json'))
    fingerprint=hashlib.sha256((ROOT/'data/db.sqlite3').read_bytes()).hexdigest()
    assert first['database_sha256']==second['database_sha256']==fingerprint
    frames=[]
    for directory,report in [(OUT,first),(second_dir,second)]:
        data=pd.read_csv(directory/'predictions.csv');data=data[data.phase=='test'].copy()
        assert (data.groupby(['method','seed']).size()==342).all()
        assert not data.duplicated(['event','rank','stage','method','seed']).any()
        assert np.isfinite(data.predicted).all() and (data.predicted>=data.current-1e-6).all()
        assert (data.observed_at<=data.cutoff).all()
        assert all(r['training_latest_end']+15*60000<r['test_earliest_start'] for r in report['training_runs'])
        frames.append(data)
    all_data=pd.concat(frames,ignore_index=True);all_data['ape']=abs(all_data.predicted/all_data.actual-1)*100
    reference=all_data[all_data.method=='lgbm_baseline'].set_index(['event','rank','stage']).sort_index()
    for _,g in all_data.groupby(['method','seed']):
        g=g.set_index(['event','rank','stage']).sort_index()
        for column in ['current','actual','cutoff','observed_at']:
            np.testing.assert_array_equal(g[column].to_numpy(),reference[column].to_numpy())
    old=pd.read_csv(ROOT/'analysis/advanced_results/predictions.csv')
    old=old[(old.phase=='test')&(old.method=='lightgbm_sequence_residual')].set_index(['event','rank','stage']).sort_index()
    np.testing.assert_array_equal(old.predicted.to_numpy(),reference.predicted.to_numpy())
    summaries=[]
    for method,label in LABELS.items():
        g=all_data[all_data.method==method];assert len(g)==342
        summaries.append({'method':method,'label':label,'n':len(g),'mape':float(g.ape.mean()),
            'rank1000_mape':float(g[g['rank']==1000].ape.mean()),
            'rank2000_mape':float(g[g['rank']==2000].ape.mean()),'p90_ape':float(g.ape.quantile(.9))})
    table=pd.DataFrame(summaries);table.to_csv(OUT/'comparison-summary.csv',index=False)
    selected=table[table.method==first['selection']['selected_recipe']].iloc[0]
    baseline=table[table.method=='lgbm_baseline'].iloc[0]
    result={'status':'passed','test_events':36,'test_scenarios':342,
        'configuration_runs':len(first['training_runs'])+len(second['training_runs']),
        'database_sha256':fingerprint,'selected_on_development':first['selection']['selected_recipe'],
        'baseline_mape':float(baseline.mape),'selected_mape':float(selected.mape),
        'relative_mape_reduction':float(1-selected.mape/baseline.mape),
        'source_metadata_policy':'current fetched metadata; CN card releases filtered by observation time; not a historical metadata archive',
        'band_definition':'composition of event bonus characters, retaining mixed bands',
        'checks':['same 342 cases','no duplicated predictions','chronological training','frozen fold historical statistics',
                  'original baseline exactly reproduced','finite predictions','source database unchanged'],
        'summary':summaries}
    (OUT/'comparison-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(table[['label','mape','rank1000_mape','rank2000_mape']].round(4).to_string(index=False))
    print('VALIDATION',json.dumps({k:v for k,v in result.items() if k!='summary'},ensure_ascii=False))


if __name__=='__main__':main()
