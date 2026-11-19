"""Separate two-rank true fitting sample-size study, validation only.

This changes the base models and is not an identical-base V3 replay. The
coordinator injects TRAIN supervision; no validation/test answers are read.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.multirank_dataset import identity
from analysis.multirank_model import feature_row,make_preprocessor
from analysis.temporal_dataset import TemporalDataset
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json

DATA=ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115'
RUN=ROOT/'analysis/paper_upgrade_records/20261118/runs/two_rank_learning_curve_v2'
DATASET_ID='d057beca6597dd28b59fde7da08f665ae74af41d96f940da04268c881864a215'
SIZES=[16,32,48,67]
SEEDS=[42,123,2026]
TREE={'iterations':200,'depth':5,'learning_rate':.035,'l2_leaf_reg':12,'random_seed':42,'thread_count':1,'verbose':False,'allow_writing_files':False}
NEURAL={'encoder':[16,16],'head':[8,1],'epochs':60,'learning_rate':.002,'weight_decay':.05,'dropout':.1,'seeds':SEEDS,'threads':1,'loss':'event-equal MSE adjusted log growth'}


def event_weights(events):
    _,inverse,count=np.unique(events,return_inverse=True,return_counts=True)
    weights=1./count[inverse];return weights/weights.mean()


def training_subsets(cases,validation_start,sizes=SIZES):
    clock=cases[['event','start_ms','end_ms']].drop_duplicates()
    if clock.event.duplicated().any() or (clock.start_ms<1706504400000).any() or (clock.end_ms>=validation_start).any():
        raise ValueError('Only completed dense training events can supervise validation')
    events=clock.sort_values(['start_ms','event']).event.astype(int).tolist()
    if any(n<=0 or n>len(events) for n in sizes):raise ValueError('Unavailable training sample size')
    return {n:events[:n] for n in sizes}


def target_values(current,progress,actual):
    current,progress,actual=map(lambda x:np.asarray(x,float),[current,progress,actual])
    if (not all(np.isfinite(x).all() for x in [current,progress,actual]) or (current<=0).any()
            or (actual<=0).any() or (progress<=0).any() or (progress>1).any()):raise ValueError('Invalid training target inputs')
    return np.log(actual/current)+np.log(progress)


def decode(adjusted_log_growth,current,progress):
    return np.asarray(current,float)*np.exp(np.clip(np.asarray(adjusted_log_growth,float)-np.log(progress),0,4))


def network(features,mode):
    import torch
    from torch import nn
    if mode not in ['shared','rank_heads']:raise ValueError('Unknown fixed network mode')
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(features,16),nn.SiLU(),nn.Dropout(.1),nn.Linear(16,16),nn.SiLU())
            def head():return nn.Sequential(nn.Linear(16,8),nn.SiLU(),nn.Dropout(.1),nn.Linear(8,1))
            self.heads=nn.ModuleList([head() for _ in range(1 if mode=='shared' else 2)])
        def forward(self,x,rank):
            z=self.encoder(x);columns=torch.cat([h(z) for h in self.heads],dim=1)
            return columns[:,0] if mode=='shared' else columns.gather(1,rank[:,None]).squeeze(1)
    return Model()


def fit_network(x,y,weights,rank,qx,qrank,*,mode,seed):
    import torch
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=network(x.shape[1],mode);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.05)
        xt=torch.tensor(x,dtype=torch.float32);qt=torch.tensor(qx,dtype=torch.float32)
        yt=torch.tensor(y,dtype=torch.float32);wt=torch.tensor(weights/weights.sum(),dtype=torch.float32)
        rt=torch.tensor(rank,dtype=torch.long);qrt=torch.tensor(qrank,dtype=torch.long);losses=[]
        for _ in range(60):
            model.train();optimizer.zero_grad(set_to_none=True);prediction=model(xt,rt)
            loss=(wt*(prediction-yt).square()).sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural curve loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():predicted=model(qt,qrt).numpy().astype(float)
        return predicted,{'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'seed':seed,'losses':losses,'features':x.shape[1],'mode':mode}


def fit_independent_networks(x,y,events,rank,qx,qrank,*,seed):
    """Two parameter-independent encoders/heads using the same prepared features.

    Every fit gets only its own rank's rows, labels and event weights. Shared
    preprocessing was fitted on the permitted training pool by the coordinator.
    """
    predicted=np.empty(len(qx));checkpoints={}
    for code,r in enumerate([1000,2000]):
        own=rank==code;query=qrank==code
        if not own.any():raise ValueError('Both independent ranks need training rows')
        weights=event_weights(np.asarray(events)[own]);wall=time.monotonic();cpu=time.process_time()
        values,checkpoint=fit_network(x[own],np.asarray(y)[own],weights,rank[own],qx[query],qrank[query],mode='shared',seed=seed)
        checkpoint['fit_wall_seconds']=time.monotonic()-wall;checkpoint['fit_cpu_seconds']=time.process_time()-cpu
        predicted[query]=values;checkpoints[r]=checkpoint
    return predicted,checkpoints


def configuration(dataset=DATA):
    dataset=Path(dataset);manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest['dataset_id']!=DATASET_ID:raise ValueError('Exact dense two-rank temporal dataset required')
    sources=['training/analysis/paper_upgrade_learning_curve.py','training/analysis/multirank_model.py','training/analysis/temporal_dataset.py','training/analysis/multirank_dataset.py','training/analysis/backtest.py','training/analysis/holdout_dataset.py']
    methods=[]
    for n in SIZES:
        methods += [f'catboost_shared_n{n}',f'catboost_independent_n{n}']
        for mode in ['shared','rank_heads','independent']:methods += [f'neural_{mode}_n{n}_seed{s}' for s in SEEDS]+[f'neural_{mode}_n{n}_mean']
    cfg={'protocol':'paper_two_rank_learning_curve_v2','dataset_id':manifest['dataset_id'],'dataset_directory':str(dataset.resolve()),
         'sizes':SIZES,'ranks':[1000,2000],'evaluation_split':'validation','evaluation_cases':488,'evaluation_events':14,
         'methods':methods,'tree_recipe':TREE,'neural_recipe':NEURAL,'tree_fits':12,'network_fits':48,
         'supervision':'earliest N training activities by actual start; all frozen prefixes, equal event mass within fitted scope',
         'features':'own multirank_model.feature_row local shape plus log_rank, same pooled preprocessing for shared/independent',
         'target':'log(final/current)+log(actual observed progress)','decoder':'current*exp(clip(adjusted prediction-log(progress),0,4))',
         'neural_normalization':'event-weighted mean/std fitted on selected training matrix; same for all three structures',
         'independent_neural':'two parameter-independent16/16 encoders and8/1 heads, each trained only on its rank with own event-equal weights; shared training-fit preprocessing; same60epochs is not same parameter count or total fit budget',
         'selection':'none; descriptive frozen validation study, no test read or selection',
         'scope':'new CatBoost200 and16/16 neural base fits; not original V3 mixed base or residual history sensitivity',
         'exposure':'Repeatedly used validation; descriptive sample-size diagnostic','metadata_caveat':'Retrospective snapshot; local own features only in this new fitting study',
         'source_hashes':{p:sha256(ROOT/p) for p in sources},'python':platform.python_version(),
         'software':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','catboost','torch','joblib']},
         'resource_policy':{'processes':1,'threads':1,'nice':19},'blind_answers':['validation_labels.csv','test_labels.csv']}
    cfg['run_id']=identity(cfg);return cfg


def prepare(run=RUN,dataset=DATA):
    run=Path(run)
    if run.exists():raise FileExistsError('Preserve old runs; choose a new directory')
    cfg=configuration(dataset);run.mkdir(parents=True);write_new_json(run/'run_manifest.json',cfg)
    write_new_json(run/'freeze_receipt.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'run_id':cfg['run_id'],'numeric_started':False})
    for name in cfg['source_hashes']:
        path=run/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    seal_directory(run,'configuration_seal.json',{'run_id':cfg['run_id'],'stage':'prepared_only'})
    return cfg


def execute(run=RUN):
    run=Path(run);cfg=json.loads((run/'run_manifest.json').read_text());verify_seal(run/'configuration_seal.json')
    if cfg!=configuration(cfg['dataset_directory']):raise ValueError('Frozen source/configuration changed')
    if (run/'models').exists():raise FileExistsError('Preserve partial/completed fitting')
    data=TemporalDataset(cfg['dataset_directory'])
    # Exactly TRAIN labels are allowed in supervised fitting; VALIDATION answers
    # stay inaccessible until a different independent scoring entry point runs.
    labels=data.labels('train');train=list(data.inputs('train'));valid=list(data.inputs('validation'))
    tc=data.case_index('train').set_index('case_id');vc=data.case_index('validation').set_index('case_id')
    def rows(samples,cases,split):
        result=[]
        for sample in samples:
            c=cases.loc[sample['case_id']].to_dict();c.update(case_id=sample['case_id'],split=split,progress=sample['progress'],phase=int(np.searchsorted([.25,.9],sample['progress'],side='right')))
            result.append(c)
        return pd.DataFrame(result)
    train_cases=rows(train,tc,'train');query=rows(valid,vc,'validation')
    if len(query)!=488 or query.event.nunique()!=14:raise ValueError('Frozen validation panel differs')
    subsets=training_subsets(train_cases,int(query.start_ms.min()));x=pd.DataFrame([feature_row(s) for s in train]);qx=pd.DataFrame([feature_row(s) for s in valid])
    actual=np.array([labels[c] for c in train_cases.case_id]);y=target_values(train_cases.current,train_cases.progress,actual)
    query.to_csv(run/'validation_features.csv',index=False);train_cases.to_csv(run/'training_cases.csv',index=False)
    write_new_json(run/'training_subsets.json',{str(n):e for n,e in subsets.items()});(run/'models').mkdir();results=[];costs=[]
    rank=train_cases['rank'].map({1000:0,2000:1}).to_numpy(int);qrank=query['rank'].map({1000:0,2000:1}).to_numpy(int)
    def record(method,predicted,n,seed,family,structure):
        if len(predicted)!=len(query) or not np.isfinite(predicted).all():raise ValueError('Missing curve prediction')
        frame=query.copy();frame['method']=method;frame['predicted']=predicted;frame['training_events']=n;frame['seed']=str(seed);frame['family']=family;frame['structure']=structure
        results.append(frame)
    # Tree and neural fits are sequential, and all objects are detached arrays.
    for n,events in subsets.items():
        began=time.monotonic();mask=train_cases.event.isin(events).to_numpy();fitting=train_cases.loc[mask]
        weights=event_weights(fitting.event.to_numpy());prep=make_preprocessor(x.loc[mask]);matrix=prep.fit_transform(x.loc[mask]);prediction_matrix=prep.transform(qx)
        path=run/'models'/f'n{n}';path.mkdir();joblib.dump(prep,path/'preprocessor.joblib',compress=3)
        fitting.assign(weight=weights).to_csv(path/'training_weights.csv',index=False)
        wall=time.monotonic();cpu=time.process_time()
        model=CatBoostRegressor(**TREE);model.fit(matrix,y[mask],sample_weight=weights);joblib.dump(model,path/'catboost_shared.joblib',compress=3)
        costs.append({'training_events':n,'family':'catboost','structure':'shared','seed':42,'rank':'both','training_rows':len(matrix),'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu,'iterations_or_epochs':200})
        record(f'catboost_shared_n{n}',decode(model.predict(prediction_matrix),query.current,query.progress),n,42,'catboost','shared')
        prediction=np.empty(len(query))
        for code,r in enumerate([1000,2000]):
            own=rank[mask]==code;qw=qrank==code;ownweights=event_weights(fitting.loc[own].event.to_numpy())
            wall=time.monotonic();cpu=time.process_time()
            model=CatBoostRegressor(**TREE);model.fit(matrix[own],y[mask][own],sample_weight=ownweights);joblib.dump(model,path/f'catboost_rank{r}.joblib',compress=3)
            costs.append({'training_events':n,'family':'catboost','structure':'independent','seed':42,'rank':r,'training_rows':int(own.sum()),'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu,'iterations_or_epochs':200})
            prediction[qw]=model.predict(prediction_matrix[qw])
        record(f'catboost_independent_n{n}',decode(prediction,query.current,query.progress),n,42,'catboost','independent')
        mean=np.average(matrix,axis=0,weights=weights);std=np.maximum(np.sqrt(np.average((matrix-mean)**2,axis=0,weights=weights)),1e-4)
        nx=np.clip((matrix-mean)/std,-8,8);nqx=np.clip((prediction_matrix-mean)/std,-8,8)
        joblib.dump({'mean':mean,'std':std},path/'neural_scaler.joblib',compress=3)
        for mode in ['shared','rank_heads']:
            members=[]
            for seed in SEEDS:
                wall=time.monotonic();cpu=time.process_time()
                raw,checkpoint=fit_network(nx,y[mask],weights,rank[mask],nqx,qrank,mode=mode,seed=seed)
                import torch
                torch.save(checkpoint,path/f'neural_{mode}_seed{seed}.pt')
                restored=network(nx.shape[1],mode)
                restored.load_state_dict(torch.load(path/f'neural_{mode}_seed{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);restored.eval()
                with torch.no_grad():replayed=restored(torch.tensor(nqx,dtype=torch.float32),torch.tensor(qrank,dtype=torch.long)).numpy()
                if not np.allclose(raw,replayed,rtol=1e-6,atol=1e-6):raise ValueError('Saved neural checkpoint prediction parity failed')
                costs.append({'training_events':n,'family':'neural','structure':mode,'seed':seed,'rank':'both','training_rows':len(nx),'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu,'iterations_or_epochs':60,'parameters':sum(p.numel() for p in restored.parameters())})
                predicted=decode(raw,query.current,query.progress);members.append(predicted)
                record(f'neural_{mode}_n{n}_seed{seed}',predicted,n,seed,'neural',mode)
            record(f'neural_{mode}_n{n}_mean',np.mean(members,axis=0),n,'mean','neural',mode)
        members=[]
        for seed in SEEDS:
            raw,checkpoints=fit_independent_networks(nx,y[mask],fitting.event.to_numpy(),rank[mask],nqx,qrank,seed=seed)
            for code,r in enumerate([1000,2000]):
                checkpoint=checkpoints[r];filename=path/f'neural_independent_rank{r}_seed{seed}.pt';torch.save(checkpoint,filename)
                restored=network(nx.shape[1],'shared');restored.load_state_dict(torch.load(filename,map_location='cpu',weights_only=True)['state_dict']);restored.eval();ownq=qrank==code
                with torch.no_grad():replayed=restored(torch.tensor(nqx[ownq],dtype=torch.float32),torch.tensor(qrank[ownq],dtype=torch.long)).numpy()
                if not np.allclose(raw[ownq],replayed,rtol=1e-6,atol=1e-6):raise ValueError('Independent checkpoint prediction parity failed')
                costs.append({'training_events':n,'family':'neural','structure':'independent','seed':seed,'rank':r,'training_rows':int((rank[mask]==code).sum()),'wall_seconds':checkpoint['fit_wall_seconds'],'cpu_seconds':checkpoint['fit_cpu_seconds'],'iterations_or_epochs':60,'parameters':sum(p.numel() for p in restored.parameters())})
            predicted=decode(raw,query.current,query.progress);members.append(predicted)
            record(f'neural_independent_n{n}_seed{seed}',predicted,n,seed,'neural','independent')
        record(f'neural_independent_n{n}_mean',np.mean(members,axis=0),n,'mean','neural','independent')
        write_new_json(path/'fit_manifest.json',{'training_events':n,'event_ids':events,'training_cases':len(fitting),'validation_case_ids_sha256':identity(query.case_id.tolist()),'fit_before_ms':int(query.start_ms.min()),'seconds':time.monotonic()-began})
        print(json.dumps({'training_events':n,'training_cases':len(fitting),'validation_cases':len(query),'tree_fits':3,'network_fits':12,'seconds':round(time.monotonic()-began,3)}),flush=True)
    forecasts=pd.concat(results,ignore_index=True)
    if set(forecasts.method)!=set(cfg['methods']):raise ValueError('Learning-curve method panel changed')
    for _,part in forecasts.groupby('method'):
        if part.case_id.duplicated().any() or set(part.case_id)!=set(query.case_id):raise ValueError('Learning-curve cases differ')
    forecasts.to_csv(run/'validation_predictions.csv',index=False)
    pd.DataFrame(costs).to_csv(run/'fit_costs.csv',index=False)
    for source,digest in cfg['source_hashes'].items():
        if sha256(ROOT/source)!=digest:raise ValueError('Source changed during fitting')
    return seal_directory(run,'blind_manifest.json',{'protocol':cfg['protocol'],'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],'dataset_directory':cfg['dataset_directory'],'scored_splits':['validation'],'methods':cfg['methods'],'test_labels_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','execute']);p.add_argument('--run',type=Path,default=RUN);a=p.parse_args()
    result=prepare(a.run) if a.action=='prepare' else execute(a.run);print(json.dumps({'run_id':result['run_id'],'action':a.action}))
