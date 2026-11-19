"""Partially pooled type effects and explicit reliability of causal curve priors."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from analysis.relationship_benchmark import prep_for, weights, target_values, decode
from analysis.deadline_benchmark import TREE_PARAMETERS

RECIPES=['remaining_rank','additive_spline','hierarchical_spline','quality_spline',
         'quality_shared_tree','quality_type_tree']
SPLINE_COLUMNS=['progress','log_remaining_h']


def quality_features(history):
    """No quality-based activity reweighting. Unknown shapes have zero deviation."""
    n=history.hist_effective_n.fillna(0).clip(lower=0)
    span=history.hist_span_h.fillna(np.inf).clip(lower=0)
    valid=history.hist_shape_logk.notna() & history.hist_span_h.notna()
    reliability=(n/(n+10)/(1+span/2)).where(valid,0)
    pooled=history.hist_deadline_logk.fillna(history.hist_progress_logk)
    return pd.DataFrame({'history_progress':history.hist_progress_logk,
        'history_deadline':history.hist_deadline_logk,
        'quality_shape_reliability':reliability,
        'quality_shape_deviation':reliability*(history.hist_shape_logk-pooled).fillna(0),
        'history_available':pooled.notna().astype(float)})


def type_shares(samples):
    groups={}
    for s in samples:groups.setdefault(s['curve'].kind,set()).add(s['curve'].event)
    return {k:len(v)/(len(v)+20) for k,v in sorted(groups.items())}


def design(bundle,frame,samples,fit=False):
    x=bundle['prep'].fit_transform(frame) if fit else bundle['prep'].transform(frame)
    if 'spline' not in bundle:return np.asarray(x)
    z=frame[SPLINE_COLUMNS]
    b=bundle['spline'].fit_transform(z) if fit else bundle['spline'].transform(z)
    pieces=[x,b]
    if bundle['recipe']!='additive_spline':
        basis=np.column_stack([np.ones(len(samples)),b])
        types=np.array([s['curve'].kind for s in samples]);ranks=np.array([s['curve'].rank for s in samples])
        for kind in bundle['types']:
            # Scaling yields explicit stronger L2 penalties on group departures.
            pieces.append(basis*(types==kind)[:,None]/2)
            for rank in [1000,2000]:pieces.append(basis*((types==kind)&(ranks==rank))[:,None]/4)
    return np.column_stack(pieces)


def fit(samples,labels,base,quality,recipe):
    frame=quality if recipe.startswith('quality_') else base
    bundle={'recipe':recipe,'columns':list(frame),'prep':prep_for(frame),
            'types':sorted({s['curve'].kind for s in samples}),'models':{}}
    if recipe.endswith('spline'):
        bundle['spline']=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
            SplineTransformer(n_knots=4,degree=2,include_bias=False,knots='uniform',extrapolation='constant'))
    x=design(bundle,frame,samples,fit=True);y=target_values(samples,labels,'remaining');w=weights(samples)
    if recipe.endswith('spline'):
        model=Ridge(alpha=20);model.fit(x,y,sample_weight=w);bundle['models']['all']=model
    else:
        ranks=np.array([s['curve'].rank for s in samples]);types=np.array([s['curve'].kind for s in samples])
        keys=[1000,2000] if recipe=='remaining_rank' else ['all']
        if recipe=='quality_type_tree':keys+=bundle['types'];bundle['shares']=type_shares(samples)
        for key in keys:
            mask=(ranks==key) if isinstance(key,int) else (types==key) if key!='all' else np.ones(len(samples),bool)
            model=CatBoostRegressor(**TREE_PARAMETERS)
            # Never independently renormalize rank/type expert weights.
            model.fit(x[mask],y[mask],sample_weight=w[mask]);bundle['models'][key]=model
    return bundle


def predict(bundle,samples,base,quality):
    frame=quality if bundle['recipe'].startswith('quality_') else base
    x=design(bundle,frame,samples);models=bundle['models'];n=len(samples)
    if 'all' in models:
        values=models['all'].predict(x)
        if bundle['recipe']=='quality_type_tree':
            kinds=np.array([s['curve'].kind for s in samples])
            for kind,share in bundle['shares'].items():
                mask=kinds==kind
                if mask.any():values[mask]=(1-share)*values[mask]+share*models[kind].predict(x[mask])
    else:
        ranks=np.array([s['curve'].rank for s in samples]);values=np.empty(n)
        for rank,model in models.items():
            mask=ranks==rank
            if mask.any():values[mask]=model.predict(x[mask])
    return decode(samples,values,'remaining')
