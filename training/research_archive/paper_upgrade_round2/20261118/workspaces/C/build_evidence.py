"""Reproduce report tables solely from sealed C scores/costs; no labels/fits."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import c_fairness as c

HERE=Path(__file__).resolve().parent

def main(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    runs=HERE/'runs'
    for name in ['c1_002','c2_002']:c.verify(runs/name,'blind_prediction_seal.json')
    for name in ['c1_002_scores','c2_002_scores']:c.verify(runs/name,'evaluation_seal.json')
    output.mkdir(parents=True)
    s1=pd.read_csv(runs/'c1_002_scores/summary.csv')
    last=s1[s1.period=='temporal_diagnostic'].copy()
    baseline=last[last.method=='independent'].set_index('panel').event_macro_mape_pct
    last['delta_vs_independent_pp']=[r.event_macro_mape_pct-baseline[r.panel] for r in last.itertuples()]
    last.to_csv(output/'C1_last10_all_ablations.csv',index=False)
    e1=pd.read_csv(runs/'c1_002_scores/by_event.csv')
    baseline=e1[e1.method=='independent'].set_index(['panel','period','event']).event_macro_mape_pct
    e1['delta_vs_independent_pp']=[r.event_macro_mape_pct-baseline[(r.panel,r.period,r.event)] for r in e1.itertuples()]
    e1.to_csv(output/'C1_paired_event_differences.csv',index=False)
    s2=pd.read_csv(runs/'c2_002_scores/summary.csv')
    s2=s2[s2.method.str.endswith('_mean')].copy()
    s2['group']=s2.method.str.extract(r'^(N32full|N16span|N16recent)_')[0]
    s2['structure']=s2.method.str.extract(r'_(shared60|shared30|independent60)_mean$')[0]
    wide=s2.pivot(index=['endpoint','group'],columns='structure',values='event_macro_mape_pct').reset_index()
    wide['shared60_minus_independent_pp']=wide.shared60-wide.independent60
    wide['shared30_minus_independent_pp']=wide.shared30-wide.independent60
    wide.to_csv(output/'C2_parameter_and_MAC_contrasts.csv',index=False)
    s2.groupby(['group','structure']).event_macro_mape_pct.mean().unstack().to_csv(output/'C2_12event_aggregate.csv')
    e2=pd.read_csv(runs/'c2_002_scores/by_event.csv');e2=e2[e2.method.str.endswith('_mean')].copy()
    e2['group']=e2.method.str.extract(r'^(N32full|N16span|N16recent)_')[0]
    e2['structure']=e2.method.str.extract(r'_(shared60|shared30|independent60)_mean$')[0]
    event=e2.pivot(index=['endpoint','group','event'],columns='structure',values='event_macro_mape_pct').reset_index()
    event['shared60_minus_independent_pp']=event.shared60-event.independent60
    event['shared30_minus_independent_pp']=event.shared30-event.independent60
    event.to_csv(output/'C2_paired_event_differences.csv',index=False)
    costs=pd.read_csv(runs/'c2_002/fit_costs.csv')
    costs.groupby('structure')[['parameters','optimizer_steps','parameter_updates','forward_linear_MACs','cpu_seconds','wall_seconds']].sum().to_csv(output/'C2_budget_totals.csv')
    # Parameter equality is checked per endpoint/group/seed, after summing both independent models.
    counts=costs.groupby(['endpoint','group','seed','structure']).parameters.sum().unstack()
    if not ((counts.shared60==counts.shared30)&(counts.shared60==counts.independent60)).all():raise ValueError('Actual parameter counts differ')
    counts.to_csv(output/'C2_actual_parameter_equality.csv')
    predictions=pd.read_csv(runs/'c1_002/predictions.csv.gz',usecols=['case_id','method','predicted'])
    values=predictions.pivot(index='case_id',columns='method',values='predicted')
    support=pd.read_csv(runs/'c1_002/supports.csv')
    checks={'same_filter_vs_same_panel_max_abs_score_difference':float(np.abs(values.same_filter_pair-values.same_panel_independent).max()),
      'same_panel_vs_independent_max_abs_score_difference':float(np.abs(values.same_panel_independent-values.independent).max()),
      'complete_support_vs_legacy_max_abs_score_difference':float(np.abs(values.complete_support-values.legacy_pair).max()),
      'maximum_anchor_minus_pair_support':float((support.anchor_events-support.paired_events).dropna().abs().max()),
      'C1_predictions':len(predictions),'C1_physical_cases':predictions.case_id.nunique(),
      'C2_physical_fits':len(costs),'shared60_win_cells':int((wide.shared60<wide.independent60).sum()),
      'shared30_win_cells':int((wide.shared30<wide.independent60).sum()),'comparison_cells':len(wide)}
    c.write_json(output/'audit_checks.json',checks)
    c.write_json(output/'evidence_sources.json',{'script_sha256':c.sha(__file__),
      'source_seals':{name:c.sha(runs/name/seal) for name,seal in [('c1_002','blind_prediction_seal.json'),('c2_002','blind_prediction_seal.json'),('c1_002_scores','evaluation_seal.json'),('c2_002_scores','evaluation_seal.json')]}})
    c.seal(output,'evidence_seal.json')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
