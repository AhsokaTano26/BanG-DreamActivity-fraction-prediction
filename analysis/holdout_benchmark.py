"""Train once on holdout_v1, select on validation, then emit blind test scores.

Run: .venv/bin/python -m analysis.holdout_benchmark --run analysis/holdout_results/v1
Score separately: .venv/bin/python -m analysis.holdout_evaluate --run analysis/holdout_results/v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import torch
from torch.nn import functional as F

from analysis.advanced_models import ARCHITECTURES, Forecaster, prefix_sequence
from analysis.backtest import ROOT, templates
from analysis.event_features import EventFeatures
from analysis.holdout_dataset import DEFAULT_DATASET, HoldoutDataset, sha256, write_json
from analysis.holdout_evaluate import case_metadata, score_predictions
from analysis.holdout_features import HoldoutFeatureBuilder

BASELINES = ['linear','recent_24h','template','same_type_template','shape_analog']
TREE_RECIPES = {
    'ridge_growth':('baseline','ridge'), 'xgb_tab_growth':('baseline','xgb_tab'),
    'xgb_sequence':('baseline','xgb'), 'extra_trees_sequence':('baseline','extra_trees'),
    'lightgbm_sequence':('baseline','lightgbm'), 'catboost_sequence':('baseline','catboost'),
    **{f'{backend}_{mode}':(mode,backend) for backend in ['lightgbm','catboost']
       for mode in ['band','static','history','peer']},
}
NEURAL_RECIPES = {name:('baseline',name) for name in ARCHITECTURES} | {'tcn_peer':('peer','tcn')}
SEEDS = [42,123,2026]
SOURCES = ['analysis/holdout_dataset.py','analysis/holdout_features.py','analysis/holdout_benchmark.py',
           'analysis/holdout_evaluate.py','analysis/advanced_models.py','analysis/backtest.py','analysis/event_features.py']


def event_weights(samples):
    events = np.array([s['curve'].event for s in samples])
    _, indices, counts = np.unique(events,return_inverse=True,return_counts=True)
    values = 1/counts[indices]
    return (values/values.mean()).astype(np.float32)


def event_mape(prediction, actual, samples):
    events = np.array([s['curve'].event for s in samples])
    _, indices, counts = np.unique(events,return_inverse=True,return_counts=True)
    return float(np.mean(np.bincount(indices,weights=abs(prediction/actual-1)*100)/counts))


def final_predictions(samples, residual):
    current = np.array([s['score'] for s in samples])
    base = -np.log(np.array([s['progress'] for s in samples]))
    return current*np.exp(np.clip(base+residual,0,4))


def baseline_predictions(samples, history):
    values = {name:[] for name in BASELINES}
    for sample in samples:
        score, p = sample['score'],sample['progress']
        linear = score/p
        rate = sample['features']['rate_24h_relative']
        recent = score + max(0.,rate)*score/sample['features']['elapsed_h']*sample['features']['remaining_h'] if np.isfinite(rate) else linear
        usable = [c for c in history if c.event != sample['curve'].event and c.end+15*60000 < sample['curve'].start]
        template, typed, shape = templates(sample,usable)
        for name, value in zip(BASELINES,[linear,recent,template,typed,shape]):
            values[name].append(float(max(score,value)))
    return {name:np.array(value) for name,value in values.items()}


def make_preprocessor(frame):
    categorical = [column for column in frame if not pd.api.types.is_numeric_dtype(frame[column])]
    numerical = [column for column in frame if column not in categorical]
    return ColumnTransformer([
        ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),numerical),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])


def fit_tree(backend, tab, sequence, growth, base, weights):
    if backend == 'ridge':
        model = Ridge(alpha=20)
    elif backend == 'xgb_tab':
        model = XGBRegressor(n_estimators=180,max_depth=2,learning_rate=.04,min_child_weight=12,
            reg_lambda=10,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=4)
    elif backend == 'xgb':
        model = XGBRegressor(n_estimators=400,max_depth=3,learning_rate=.03,min_child_weight=12,
            reg_lambda=10,subsample=.9,colsample_bytree=.8,random_state=42,n_jobs=4)
    elif backend == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,max_features=.8,
                                    max_depth=12,random_state=42,n_jobs=4)
    elif backend == 'lightgbm':
        model = LGBMRegressor(n_estimators=500,max_depth=4,num_leaves=15,learning_rate=.03,
                             min_child_samples=30,reg_lambda=10,random_state=42,n_jobs=4,verbosity=-1)
    elif backend == 'catboost':
        model = CatBoostRegressor(iterations=600,depth=4,learning_rate=.03,l2_leaf_reg=10,
            random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    else:
        raise ValueError(backend)
    tab_only = backend in ['ridge','xgb_tab']
    x = tab if tab_only else np.concatenate([tab,sequence.reshape(len(tab),-1)],axis=1)
    model.fit(x,growth if tab_only else growth-base,sample_weight=weights)
    return {'model':model,'tab_only':tab_only,'growth_target':tab_only}


def tree_predict(bundle, tab, sequence, base):
    x = tab if bundle['tab_only'] else np.concatenate([tab,sequence.reshape(len(tab),-1)],axis=1)
    values = bundle['model'].predict(x)
    return values-base if bundle['growth_target'] else values


def tensor_predict(model, sequence, tab, batch_size):
    return np.concatenate([model(sequence[i:i+batch_size],tab[i:i+batch_size]).detach().cpu().numpy()
                           for i in range(0,len(tab),batch_size)])


def fit_neural(architecture, arrays, samples, actual, weights, *, seed, device, max_epochs, batch_size, checkpoint):
    torch.manual_seed(seed)
    if device == 'mps':torch.mps.manual_seed(seed)
    model = Forecaster(architecture,arrays['train_tab'].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    tensors = {key:torch.from_numpy(arrays[key]).to(device) for key in ['train_tab','train_seq','train_y','validation_tab','validation_seq']}
    weight = torch.from_numpy(weights).to(device)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float('inf'),0,None
    trace = []; started = time.perf_counter()
    for epoch in range(1,max_epochs+1):
        model.train(); losses = []
        order = torch.randperm(len(weight),generator=generator).to(device)
        for i in range(0,len(order),batch_size):
            index = order[i:i+batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(tensors['train_seq'][index],tensors['train_tab'][index])
            loss = (F.smooth_l1_loss(pred,tensors['train_y'][index],beta=.2,reduction='none')*weight[index]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(loss.detach())
        model.eval()
        with torch.no_grad():
            residual = tensor_predict(model,tensors['validation_seq'],tensors['validation_tab'],batch_size)
        metric = event_mape(final_predictions(samples,residual),actual,samples)
        trace.append({'epoch':epoch,'validation_event_mape_pct':metric,'train_loss':float(torch.stack(losses).mean().item())})
        if metric < best:
            best,best_epoch = metric,epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(json.dumps({'training':architecture,'seed':seed,'epoch':epoch,'best_validation_mape':round(best,4)}),flush=True)
        if epoch >= 25 and epoch-best_epoch >= 18:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        prediction = tensor_predict(model,tensors['validation_seq'],tensors['validation_tab'],batch_size)
    torch.save({'architecture':architecture,'tab_features':arrays['train_tab'].shape[1],
                'state_dict':best_state,'seed':seed,'selected_epoch':best_epoch},checkpoint)
    record = {'architecture':architecture,'seed':seed,'device':device,
              'parameters':sum(p.numel() for p in model.parameters()),'epochs_run':epoch,
              'selected_epoch':best_epoch,'validation_event_mape_pct':best,'trace':trace,
              'seconds':time.perf_counter()-started}
    del model,optimizer,tensors,weight
    if device == 'mps':torch.mps.empty_cache()
    return prediction,record


def neural_predict(checkpoint, tab, sequence, device, batch_size):
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model = Forecaster(saved['architecture'],saved['tab_features']).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    with torch.no_grad():
        result = tensor_predict(model,torch.from_numpy(sequence).to(device),torch.from_numpy(tab).to(device),batch_size)
    del model
    if device == 'mps':torch.mps.empty_cache()
    return result


def prediction_records(samples, name, prediction):
    if len(samples) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError(f'{name}: invalid prediction array')
    return [{'method':name,'case_id':s['case_id'],'event':s['curve'].event,'rank':s['curve'].rank,
             'stage':s['stage'],'event_type':s['curve'].kind,'cutoff':s['cutoff'],'observed_at':s['now'],
             'current':s['score'],'predicted':float(p)} for s,p in zip(samples,prediction)]


def run_benchmark(args):
    data = HoldoutDataset(args.dataset)
    device = 'mps' if args.device == 'auto' and torch.backends.mps.is_available() else ('cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(4)
    config = {'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),
              'device':device,'max_epochs':args.max_epochs,'batch_size':args.batch_size,'seeds':SEEDS,
              'baselines':BASELINES,'tree_recipes':TREE_RECIPES,'neural_recipes':NEURAL_RECIPES,
              'training_weights':'equal total weight per event',
              'neural_optimizer':'AdamW lr=.001 weight_decay=.001; Huber beta=.2; clip norm=1',
              'checkpoint':'minimum validation event macro MAPE; min25 epochs/patience18',
              'selection':'minimum validation event macro MAPE; top3 arithmetic final-score ensemble also reported',
              'software':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost','torch']},
              'python':platform.python_version(),'source_hashes':{name:sha256(ROOT/name) for name in SOURCES}}
    run_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run = args.run
    if run.exists():
        if not args.resume:raise FileExistsError(f'Use another run directory or --resume: {run}')
        old = json.loads((run/'run_manifest.json').read_text())
        if old['run_id'] != run_id:raise ValueError('Cannot resume with changed code, data, software, or configuration')
        if (run/'blind_test_manifest.json').exists():
            print('Blind test predictions already sealed; use the separate evaluator.');return
    else:
        run.mkdir(parents=True)
        write_json(run/'run_manifest.json',config | {'run_id':run_id})
        (run/'sources').mkdir()
        for name in SOURCES:(run/'sources'/Path(name).name).write_text((ROOT/name).read_text())
    (run/'models').mkdir(exist_ok=True);(run/'runs').mkdir(exist_ok=True)
    # No test labels or full validation/test curves are loaded here.
    train, validation = data.inputs('train'),data.inputs('validation')
    train_labels, validation_labels = data.labels('train'),data.labels('validation')
    growth = np.array([np.log(max(1.,train_labels[s['case_id']]/s['score'])) for s in train],dtype=np.float32)
    base = -np.log(np.array([s['progress'] for s in train],dtype=np.float32))
    actual = np.array([validation_labels[s['case_id']] for s in validation])
    weights = event_weights(train)
    history = data.train_curves()
    builder = HoldoutFeatureBuilder(history,EventFeatures.load(data.metadata_directory))
    print(json.dumps({'run_id':run_id,'train_events':len({s['curve'].event for s in train}),
                      'train_cases':len(train),'validation_cases':len(validation),'device':device}),flush=True)
    sequences = {split:np.stack([prefix_sequence(s) for s in samples]) for split,samples in [('train',train),('validation',validation)]}
    arrays_by_mode = {}
    for mode in ['baseline','band','static','history','peer']:
        frame = builder.frame(train,mode);prep = make_preprocessor(frame)
        arrays_by_mode[mode] = {'train_tab':prep.fit_transform(frame).astype(np.float32),
            'validation_tab':prep.transform(builder.frame(validation,mode)).astype(np.float32),
            'train_seq':sequences['train'],'validation_seq':sequences['validation'],'train_y':growth-base}
        joblib.dump(prep,run/f'models/preprocessor_{mode}.joblib')
        write_json(run/f'models/features_{mode}.json',{'columns':list(frame),'fitted_events':data.manifest['splits']['train']['event_ids']})
        print(json.dumps({'prepared_mode':mode,'features':arrays_by_mode[mode]['train_tab'].shape[1]}),flush=True)
    val_values = baseline_predictions(validation,history)
    candidate_values = dict(val_values)
    cases = case_metadata(validation)
    def save_run(key,prediction,record):
        np.save(run/f'runs/{key}.npy',prediction)
        write_json(run/f'runs/{key}.json',record | {'run_id':run_id,
            'prediction_sha256':sha256(run/f'runs/{key}.npy')})
    for name,(mode,backend) in TREE_RECIPES.items():
        a = arrays_by_mode[mode];started = time.perf_counter();key=name
        checkpoint = run/f'models/{key}.joblib'
        if args.resume and (run/f'runs/{key}.json').exists():
            record = json.loads((run/f'runs/{key}.json').read_text())
            if record['checkpoint_sha256'] != sha256(checkpoint) or record['prediction_sha256'] != sha256(run/f'runs/{key}.npy'):raise ValueError('Checkpoint cache hash mismatch')
            pred = np.load(run/f'runs/{key}.npy')
        else:
            bundle = fit_tree(backend,a['train_tab'],a['train_seq'],growth,base,weights)
            joblib.dump(bundle,checkpoint)
            residual = tree_predict(joblib.load(checkpoint),a['validation_tab'],a['validation_seq'],-np.log([s['progress'] for s in validation]))
            pred = final_predictions(validation,residual)
            record = {'method':name,'backend':backend,'mode':mode,'seconds':time.perf_counter()-started,
                      'checkpoint_sha256':sha256(checkpoint)}
            save_run(key,pred,record)
        val_values[name] = pred;candidate_values[name] = pred
        print(json.dumps({'completed':name,'validation_event_mape':round(event_mape(pred,actual,validation),4),'seconds':round(record['seconds'],2)}),flush=True)
    for name,(mode,architecture) in NEURAL_RECIPES.items():
        a = arrays_by_mode[mode];values=[]
        for seed in SEEDS:
            key=f'{name}_seed{seed}';checkpoint=run/f'models/{key}.pt'
            if args.resume and (run/f'runs/{key}.json').exists():
                record=json.loads((run/f'runs/{key}.json').read_text())
                if record['checkpoint_sha256'] != sha256(checkpoint) or record['prediction_sha256'] != sha256(run/f'runs/{key}.npy'):raise ValueError('Checkpoint cache hash mismatch')
                pred=np.load(run/f'runs/{key}.npy')
            else:
                residual,record=fit_neural(architecture,a,validation,actual,weights,seed=seed,device=device,
                    max_epochs=args.max_epochs,batch_size=args.batch_size,checkpoint=checkpoint)
                # Reload before saving predictions; the test path uses this exact checkpoint.
                reloaded=neural_predict(checkpoint,a['validation_tab'],a['validation_seq'],device,args.batch_size)
                if not np.allclose(residual,reloaded,atol=1e-5):raise ValueError('Neural checkpoint replay mismatch')
                pred=final_predictions(validation,reloaded)
                record.update({'method':name,'mode':mode,'neural':True,'checkpoint_sha256':sha256(checkpoint)})
                save_run(key,pred,record)
            val_values[key]=pred;values.append(pred)
            print(json.dumps({'completed':key,'validation_event_mape':round(event_mape(pred,actual,validation),4),'selected_epoch':record['selected_epoch'],'seconds':round(record['seconds'],2)}),flush=True)
        ensemble=np.mean(values,axis=0)
        val_values[name+'_ensemble3']=ensemble;candidate_values[name+'_ensemble3']=ensemble
    scores={name:event_mape(p,actual,validation) for name,p in candidate_values.items()}
    top3=sorted(scores,key=lambda n:(scores[n],n))[:3]
    val_values['validation_top3_ensemble']=np.mean([candidate_values[n] for n in top3],axis=0)
    scores['validation_top3_ensemble']=event_mape(val_values['validation_top3_ensemble'],actual,validation)
    selected=min(scores,key=lambda n:(scores[n],n))
    selection={'run_id':run_id,'dataset_id':data.manifest['dataset_id'],'primary_metric':'event_macro_mape_pct',
               'selected_recipe':selected,'top3_members':top3,'validation_scores':scores,
               'reported_methods':sorted(val_values),'test_labels_used_for_selection':False}
    write_json(run/'selection.json',selection)
    val_rows=[row for name,pred in val_values.items() for row in prediction_records(validation,name,pred)]
    pd.DataFrame(val_rows).to_csv(run/'validation_predictions.csv',index=False)
    val_metrics={name:score_predictions(cases,validation_labels,prediction_records(validation,name,pred)) for name,pred in val_values.items()}
    write_json(run/'validation_metrics.json',val_metrics)
    print(json.dumps({'selection_sealed':selected,'validation_mape':scores[selected],'ensemble_members':top3}),flush=True)
    # All model and preprocessor checkpoints are final before test inputs load.
    test=data.inputs('test');test_sequence=np.stack([prefix_sequence(s) for s in test])
    test_base=-np.log(np.array([s['progress'] for s in test]))
    test_tab={mode:joblib.load(run/f'models/preprocessor_{mode}.joblib').transform(builder.frame(test,mode)).astype(np.float32)
              for mode in arrays_by_mode}
    test_values=baseline_predictions(test,history)
    for name,(mode,backend) in TREE_RECIPES.items():
        residual=tree_predict(joblib.load(run/f'models/{name}.joblib'),test_tab[mode],test_sequence,test_base)
        test_values[name]=final_predictions(test,residual)
    for name,(mode,architecture) in NEURAL_RECIPES.items():
        values=[]
        for seed in SEEDS:
            key=f'{name}_seed{seed}'
            residual=neural_predict(run/f'models/{key}.pt',test_tab[mode],test_sequence,device,args.batch_size)
            test_values[key]=final_predictions(test,residual);values.append(test_values[key])
        test_values[name+'_ensemble3']=np.mean(values,axis=0)
    test_values['validation_top3_ensemble']=np.mean([test_values[n] for n in top3],axis=0)
    test_rows=[row for name,pred in test_values.items() for row in prediction_records(test,name,pred)]
    pd.DataFrame(test_rows).to_csv(run/'test_predictions.csv',index=False)
    artifacts={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.is_file()}
    write_json(run/'blind_test_manifest.json',{'run_id':run_id,'dataset_id':data.manifest['dataset_id'],
               'dataset_directory':str(data.root.resolve()),'artifacts':artifacts,'test_cases':len(test)})
    print(json.dumps({'blind_predictions_sealed':len(test_values),'cases_per_method':len(test),'test_labels_read':False}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--max-epochs',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.max_epochs<1 or args.batch_size<1:parser.error('epochs and batch size must be positive')
    run_benchmark(args)


if __name__ == '__main__':main()
