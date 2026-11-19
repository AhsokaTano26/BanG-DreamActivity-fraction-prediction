"""Post-hoc descriptive distribution diagnostics; not used to modify trained models."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from analysis.holdout_dataset import ROOT,HoldoutDataset,write_json,sha256
from analysis.event_features import EventFeatures
from analysis.quality_irregular_benchmark import frames,verify_run
from analysis.quality_hierarchy import design
from analysis.relationship_benchmark import weights


def main():
    root=ROOT/'analysis/quality_irregular_results';run=root/'v1';data=HoldoutDataset();verify_run(run,data)
    train=data.inputs('train');val=data.inputs('validation');meta=EventFeatures.load(data.metadata_directory)
    history=data.train_curves();train_base,train_quality=frames(train,history,meta);val_base,val_quality=frames(val,history,meta)
    distributions=[]
    for column in train_base.select_dtypes(include='number'):
        a=train_base[column].to_numpy(float);b=val_base[column].to_numpy(float)
        if not np.isfinite(a).any():continue
        lo,hi=np.nanmin(a),np.nanmax(a)
        distributions.append({'feature':column,'train_median':float(np.nanmedian(a)),
            'validation_median':float(np.nanmedian(b)) if np.isfinite(b).any() else np.nan,
            'train_min':float(lo),'train_max':float(hi),
            'validation_outside_training_range_cases':int(((b<lo)|(b>hi)).sum()),
            'train_missing_cases':int(np.isnan(a).sum()),'validation_missing_cases':int(np.isnan(b).sum())})
    pd.DataFrame(distributions).to_csv(root/'feature_distributions.csv',index=False)
    model=joblib.load(run/'models/final_quality_spline.joblib')
    x=design(model,train_quality,train);z=design(model,val_quality,val);coef=model['models']['all'].coef_
    names=list(model['prep'].get_feature_names_out());names += [f'spline_or_group_{i}' for i in range(len(coef)-len(names))]
    shift=np.average(z,axis=0,weights=weights(val))-np.average(x,axis=0,weights=weights(train))
    result=pd.DataFrame({'term':names,'coefficient':coef,'activity_weighted_feature_mean_change':shift,
                         'predicted_log_rate_mean_change_contribution':shift*coef})
    result.iloc[np.argsort(-abs(shift*coef))].to_csv(root/'spline_mean_shift_decomposition.csv',index=False)
    write_json(root/'distribution_diagnostics.json',{'run_id':json.loads((run/'run_manifest.json').read_text())['run_id'],
        'post_hoc_only':True,'caveat':'Train and validation differ in activity cohorts and prefix stages. Feature contributions explain the fitted model mean, not causal effects or MAPE changes.',
        'weighted_mean_prediction_change':float(shift@coef),'source_sha256':sha256(Path(__file__))})
    print(result.iloc[np.argsort(-abs(shift*coef))].head(8).to_string(index=False))


if __name__=='__main__':main()
