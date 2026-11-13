"""Within-curve controlled thinning, using training curves only."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.holdout_dataset import HoldoutDataset, sha256, write_json
from analysis.deadline_diagnostics import at_time


def run(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);data=HoldoutDataset();rows=[]
    for curve in data.train_curves():
        now=curve.end-6*3600000;original,span=at_time(curve,now)
        if not np.isfinite(original) or span>2 or len(curve.times)<100:continue
        original_share=100*(1-original/curve.final)
        for stride in [1,4,16]:
            indices=np.unique(np.r_[np.arange(0,len(curve.times),stride),len(curve.times)-1])
            thin=replace(curve,times=curve.times[indices],scores=curve.scores[indices])
            value,gap=at_time(thin,now);share=100*(1-value/thin.final)
            rows.append({'event':curve.event,'rank':curve.rank,'stride':stride,'points_before':len(curve.times),
                         'points_after':len(indices),'span_before_h':span,'span_after_h':gap,
                         'share_before_pct':original_share,'share_after_pct':share,'bias_points':share-original_share,
                         'absolute_error_points':abs(share-original_share)})
    frame=pd.DataFrame(rows);frame.to_csv(output/'controlled_thinning.csv',index=False);summary=[]
    for stride,g in frame.groupby('stride'):
        event=g.groupby('event')[['bias_points','absolute_error_points']].mean();rng=np.random.default_rng(42)
        draws=rng.integers(0,len(event),(5000,len(event)))
        summary.append({'stride':int(stride),'events':len(event),'curves':len(g),
                        'mean_bias_points':float(event.bias_points.mean()),
                        'mean_bias_event_bootstrap_95ci':np.quantile(event.bias_points.to_numpy()[draws].mean(1),[.025,.975]).tolist(),
                        'mean_absolute_error_points':float(event.absolute_error_points.mean()),
                        'median_span_after_h':float(g.span_after_h.median()),
                        'underestimation_curves':int((g.bias_points<0).sum())})
    result={'dataset_id':data.manifest['dataset_id'],'scope':'training only; same curves and endpoint, original versus subsampled observations',
            'eligibility':'at least100 observations; interpolation bracket at end-6h <=2h',
            'summary':summary,'limits':['original interpolation is an estimate, not dense ground truth',
                                        'restricted to observed dense training curves', 'no causal claim about player behavior']}
    write_json(output/'sensitivity.json',result);(output/'source.py').write_text(Path(__file__).read_text())
    write_json(output/'manifest.json',{'dataset_id':data.manifest['dataset_id'],'source_sha256':sha256(Path(__file__)),
               'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
