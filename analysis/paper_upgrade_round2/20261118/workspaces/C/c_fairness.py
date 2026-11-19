"""C isolated preregistered experiment. Run this file directly from repo root.

prepare-c1/prepare-c2 never fit. execute-* require a PI authorization receipt.
score-* are separate processes opening TRAIN answers only after prediction seal.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sys.path.insert(0,str(ROOT))
from analysis import paper_upgrade_replay_models as exact
from analysis import rolling_residual_stats as stats

MIN_START=1706504400000
DATA=ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115'
INPUTS=HERE.parents[1]/'inputs'
DATA_ID='d057beca6597dd28b59fde7da08f665ae74af41d96f940da04268c881864a215'
TRAIN_FILES=['train_inputs.jsonl.gz','train_cases.csv','train_labels.csv']
SEEDS=[42,123,2026]
C1_METHODS=['uncorrected','independent','legacy_pair','legacy_only_T1000',
 'legacy_only_T2000','swapped_only_T2000','pair_anchor_only','pair_difference_only',
 'same_filter_pair','same_panel_independent','equal_alpha','equal_shrinkage',
 'complete_support','no_anchor_shrink','no_difference_shrink','no_shrink',
 'anchor_alpha_0p3','difference_alpha_0p3']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def seal(run,name):
    run=Path(run)
    value={'files':{str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()},
           'sealed_utc':datetime.now(timezone.utc).isoformat()}
    write_json(run/name,value);return value

def verify(run,name):
    run=Path(run);value=json.loads((run/name).read_text())
    for p,digest in value['files'].items():
        if sha(run/p)!=digest:raise ValueError('Sealed file changed: '+p)
    return value

def input_hashes(inputs=INPUTS):
    return {name:sha(Path(inputs)/name) for name in ['train_residuals.csv.gz','train_queries.json']}

def verify_data_hashes(cfg):
    for path,digest in cfg['data_hashes'].items():
        if sha(path)!=digest:raise ValueError('Frozen data changed: '+path)

def c1_predict(history,current,*,query_start_ms):
    """Pure exact-recipe ablations; current is outcome-free and allowlisted."""
    past,target,_=exact.validate(history,current,query_start_ms)
    means=stats._event_means(past);warm=exact._warm(means,target)
    values={'uncorrected':target.base_prediction.to_numpy(float),
            'independent':exact.final(target,warm)}
    details={'independent':[{'rank':int(rank),'phase':int(phase),'events':int(len(part)),
                             'alpha':.3,'shrinkage':10.} for (rank,phase),part in means.groupby(['rank','phase'])]}
    legacy,details['legacy_pair']=exact.pair_bias(means,target,warm)
    values['legacy_pair']=exact.final(target,legacy)
    for name,change in [('legacy_only_T1000',1000),('legacy_only_T2000',2000)]:
        bias=warm.copy();mask=target['rank'].to_numpy()==change;bias[mask]=legacy[mask]
        values[name]=exact.final(target,bias);details[name]=details['legacy_pair']
    swapped,details['swapped_only_T2000']=exact.pair_bias(means,target,warm,recipe={'target_rank':2000,'anchor_rank':1000})
    mask=target['rank'].to_numpy()==2000;bias=warm.copy();bias[mask]=swapped[mask]
    values['swapped_only_T2000']=exact.final(target,bias)
    specs=[('pair_anchor_only','anchor_only',{}),('pair_difference_only','difference_only',{}),
      ('same_filter_pair','same_filter',{}),('same_panel_independent','same_panel',{}),
      ('equal_alpha','complete',{'anchor_alpha':.3,'difference_alpha':.3}),
      ('equal_shrinkage','complete',{'anchor_shrinkage':10.,'difference_shrinkage':10.}),
      ('no_anchor_shrink','complete',{'anchor_shrinkage':0.}),
      ('no_difference_shrink','complete',{'difference_shrinkage':0.}),
      ('no_shrink','complete',{'anchor_shrinkage':0.,'difference_shrinkage':0.}),
      ('anchor_alpha_0p3','complete',{'anchor_alpha':.3}),
      ('difference_alpha_0p3','complete',{'difference_alpha':.3})]
    for name,mode,recipe in specs:
        bias,details[name]=exact.pair_bias(means,target,warm,recipe=recipe,mode=mode)
        values[name]=exact.final(target,bias)
    # Same recipe except anchor observations also restricted to complete pairs.
    eligible=means[means['rank'].isin([1000,2000])].groupby(['event','phase'])['rank'].nunique()
    complete=set(eligible[eligible==2].index)
    paired=means.loc[[r['rank'] not in [1000,2000] or (r.event,r.phase) in complete for _,r in means.iterrows()]]
    bias,details['complete_support']=exact.pair_bias(paired,target,warm)
    values['complete_support']=exact.final(target,bias)
    if set(values)!=set(C1_METHODS):raise ValueError('C1 method menu differs')
    for name,value in values.items():
        if len(value)!=len(target) or not np.isfinite(value).all():raise ValueError('Incomplete forecast '+name)
    return values,details

def training_groups(clock,*,endpoints=(32,44,56),query_count=4):
    """Freeze actual-time sorted disjoint query blocks and three history designs."""
    rows=clock[['event','start_ms','end_ms']].drop_duplicates().sort_values(['start_ms','event'])
    if rows.event.duplicated().any() or (rows.start_ms<MIN_START).any():raise ValueError('Invalid dense event clock')
    events=rows.event.astype(int).tolist();out=[]
    for endpoint in endpoints:
        if endpoint<32 or endpoint+query_count>len(events):raise ValueError('Unavailable training endpoint')
        history=rows.iloc[endpoint-32:endpoint];query=rows.iloc[endpoint:endpoint+query_count]
        start=int(query.start_ms.min())
        if (history.end_ms>=start).any():raise ValueError('History must be strictly completed before query block')
        selected=history.event.astype(int).tolist();span=[selected[i] for i in [0,2,4,6,8,10,12,14,17,19,21,23,25,27,29,31]]
        groups={'N32full':selected,'N16span':span,'N16recent':selected[-16:]}
        out.append({'endpoint':int(endpoint),'query_events':query.event.astype(int).tolist(),
                    'query_start_ms':start,'groups':groups})
    return out

def event_rank_weights(events,ranks):
    frame=pd.DataFrame({'event':events,'rank':ranks})
    if frame.empty or frame.isna().any().any():raise ValueError('Nonempty event/rank training rows required')
    count=frame.groupby(['event','rank']).event.transform('size').to_numpy(float)
    nrank=frame.groupby('event')['rank'].transform('nunique').to_numpy(float)
    w=1./(nrank*count);return w/w.mean()

def network(d,structure):
    import torch
    from torch import nn
    if d<=0 or structure not in ['shared','independent']:raise ValueError('Invalid network dimensions/structure')
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if structure=='shared':
                self.encoder=nn.Sequential(nn.Linear(d,32,bias=False),nn.SiLU())
                self.heads=nn.ModuleList([nn.Linear(32,1) for _ in range(2)])
            else:self.models=nn.ModuleList([nn.Sequential(nn.Linear(d,16),nn.SiLU(),nn.Linear(16,1)) for _ in range(2)])
        def forward(self,x,rank):
            if x.ndim!=2 or len(rank)!=len(x) or torch.any((rank<0)|(rank>1)):raise ValueError('Invalid rank routing')
            if structure=='shared':
                z=self.encoder(x);all_heads=torch.cat([h(z) for h in self.heads],dim=1)
                return all_heads.gather(1,rank[:,None]).squeeze(1)
            out=torch.empty(len(x),dtype=x.dtype,device=x.device)
            for code,model in enumerate(self.models):
                mask=rank==code
                if torch.any(mask):out[mask]=model(x[mask]).squeeze(1)
            return out
    result=Network()
    if sum(p.numel() for p in result.parameters())!=32*d+66:raise ValueError('Parameter equality failed')
    return result

def compute_budget(d,n,epochs,structure,*,rank_rows=None):
    params=32*d+66
    if structure=='shared':mac=n*epochs*(32*d+64);steps=epochs;updates=params*epochs;fits=1
    elif structure=='independent':
        if rank_rows is None or len(rank_rows)!=2 or sum(rank_rows)!=n or min(rank_rows)<=0:raise ValueError('Both rank row counts required')
        mac=n*epochs*(16*d+16);steps=2*epochs;updates=params*epochs;fits=2
    else:raise ValueError('Unknown structure')
    return {'parameters':params,'forward_linear_MACs':int(mac),'optimizer_steps':int(steps),
            'parameter_updates':int(updates),'physical_fits':fits,'epochs':epochs,
            'MAC_definition':'linear-layer multiply-accumulate only for training forward; excludes backward, SiLU, bias and optimizer',
            'budget_caveat':'wall-clock and all optimization work are not matched'}

def allowed_train_labels(path,allowed):
    """Trusted coordinator scans TRAIN labels, materializes permitted cases only."""
    labels={};allowed=set(allowed)
    if Path(path).name!='train_labels.csv':raise ValueError('Only TRAIN supervision is allowed')
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            if row['case_id'] in allowed:
                if row['case_id'] in labels:raise ValueError('Duplicate training answer')
                value=float(row['actual'])
                if not np.isfinite(value) or value<=0:raise ValueError('Invalid past label')
                labels[row['case_id']]=value
    if set(labels)!=allowed:raise ValueError('Missing permitted training labels')
    return labels

def source_hashes():
    paths=[Path(__file__),HERE/'preregistration.json',ROOT/'analysis/paper_upgrade_replay_models.py',
      ROOT/'analysis/rolling_residual_stats.py',ROOT/'analysis/multirank_model.py',
      ROOT/'analysis/backtest.py',ROOT/'analysis/holdout_dataset.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def _prepare(run,experiment,data_hashes,**extra):
    run=Path(run)
    if run.exists():raise FileExistsError('Preserve all old/partial runs: '+str(run))
    config={'experiment':experiment,'source_hashes':source_hashes(),'data_hashes':data_hashes,
            'preregistration_sha256':sha(HERE/'preregistration.json'),
            'software':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','torch','joblib']},
            'resource_policy':{'processes':1,'threads':1,'nice':19},**extra}
    run.mkdir(parents=True);write_json(run/'configuration.json',config)
    for source in config['source_hashes']:
        dest=run/'sources'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,dest)
    seal(run,'configuration_seal.json');return config

def prepare_c2(run,dataset=DATA,*,verify_frozen=True):
    dataset=Path(dataset);cases=pd.read_csv(dataset/'train_cases.csv')
    groups=training_groups(cases)
    if verify_frozen and (len(cases)!=2452 or cases.event.nunique()!=67 or set(cases['rank'])!={1000,2000}):raise ValueError('Dense train panel differs')
    files=TRAIN_FILES if verify_frozen else ['train_cases.csv']
    hashes={str((dataset/name).resolve()):sha(dataset/name) for name in files}
    return _prepare(run,'C2',hashes,dataset_directory=str(dataset.resolve()),dataset_id=DATA_ID,
                    groups=groups,physical_fit_cap=108,seeds=SEEDS,
                    contrasts={'parameter_matched':'shared60 vs independent60',
                               'equal_forward_mac_approx':'shared30 vs independent60'})

def prepare_c1(run,inputs=INPUTS):
    inputs=Path(inputs);prereg=json.loads((HERE/'preregistration.json').read_text())
    if input_hashes(inputs)!=prereg['C1']['source_sha256']:raise ValueError('Frozen C1 input differs')
    return _prepare(run,'C1',{str((inputs/name).resolve()):sha(inputs/name) for name in ['train_residuals.csv.gz','train_queries.json']},
                    inputs_directory=str(inputs.resolve()),methods=C1_METHODS,physical_fit_cap=0)

def authorization(run,receipt,experiment):
    run=Path(run);verify(run,'configuration_seal.json');cfg=json.loads((run/'configuration.json').read_text())
    if cfg['experiment']!=experiment or cfg['source_hashes']!=source_hashes():raise ValueError('Frozen C source changed')
    verify_data_hashes(cfg)
    grant=json.loads(Path(receipt).read_text())
    if (grant.get('authorized_by')!='PI' or grant.get('experiment')!=experiment
        or grant.get('run_directory')!=str(run.resolve()) or grant.get('configuration_sha256')!=sha(run/'configuration.json')
        or grant.get('maximum_physical_fits',-1)<cfg['physical_fit_cap']):raise ValueError('PI numeric authorization does not cover frozen run')
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    return cfg

def release_query(records,*,event,start_ms):
    """Trusted coordinator boundary; never expose current outcomes to method."""
    history=records.loc[records.end_ms<int(start_ms),exact.TARGET_COLUMNS+['log_error']].copy(deep=True)
    target=records.loc[records.event==int(event),exact.TARGET_COLUMNS].copy(deep=True)
    if target.empty or (target.start_ms!=start_ms).any():raise ValueError('Query clock differs')
    return history,target

def load_c1_records(inputs):
    return pd.read_csv(Path(inputs)/'train_residuals.csv.gz',dtype={'event_type':str,'band_group':str},low_memory=False)

def seal_query(run,prediction_file,history,target,*,position):
    """Seal this issuance before moving to the next query's released outcomes."""
    start=int(target.start_ms.iloc[0])
    if (history.end_ms>=start).any():raise ValueError('Illegal query seal feedback clock')
    digest=lambda values:hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()
    receipt={'query_position':position,'event':int(target.event.iloc[0]),'query_start_ms':start,
      'max_history_end_ms':int(history.end_ms.max()) if len(history) else None,
      'history_cases':len(history),'history_events':int(history.event.nunique()),'target_cases':len(target),
      'target_case_ids_sha256':digest(target.case_id.tolist()),
      'history_case_ids_sha256':digest(history.case_id.tolist()),
      'history_event_ids_sha256':digest(list(map(int,history.event.unique()))),
      'prediction_file':str(Path(prediction_file).relative_to(run)),'prediction_sha256':sha(prediction_file),
      'sealed_utc':datetime.now(timezone.utc).isoformat()}
    write_json(Path(prediction_file).with_suffix('.issuance_seal.json'),receipt);return receipt

def check_panel(frame,expected):
    if frame.case_id.duplicated().any() or set(frame.case_id)!=set(expected):raise ValueError('Method case panel changed')
    if not np.isfinite(frame.predicted.to_numpy(float)).all():raise ValueError('Nonfinite prediction')

def metric_summary(frame):
    if frame.empty or frame.case_id.duplicated().any():raise ValueError('Invalid score panel')
    actual=frame.actual.to_numpy(float);pred=frame.predicted.to_numpy(float)
    if not np.isfinite(actual).all() or (actual<=0).any() or not np.isfinite(pred).all():raise ValueError('Invalid score values')
    error=pred-actual;ape=np.abs(error)/actual*100.
    scored=frame[['event','rank']].assign(ape=ape)
    return {'events':int(frame.event.nunique()),'cases':int(len(frame)),
      'event_macro_mape_pct':float(scored.groupby('event').ape.mean().mean()),
      'event_rank_macro_mape_pct':float(scored.groupby(['event','rank']).ape.mean().groupby('event').mean().mean()),
      'case_mape_pct':float(ape.mean()),'mae':float(np.abs(error).mean()),
      'rmse':float(np.sqrt(np.mean(error**2))),'p90_ape_pct':float(np.quantile(ape,.9))}

def load_train_prefixes(dataset):
    """Input-only loader; no Dataset constructor, metadata, peers or label read."""
    from analysis.holdout_dataset import from_record
    from analysis.backtest import snapshot
    from analysis.multirank_model import feature_row
    dataset=Path(dataset);cases=pd.read_csv(dataset/'train_cases.csv').set_index('case_id',drop=False)
    if cases.index.duplicated().any():raise ValueError('Duplicate TRAIN case')
    features=[];rows=[];seen=set()
    with gzip.open(dataset/'train_inputs.jsonl.gz','rt') as f:
        for line in f:
            record=json.loads(line);curve=from_record(record)
            if record['case_id'] in seen or not record['case_id'].startswith('train:'):raise ValueError('Invalid TRAIN identity')
            if curve.start<MIN_START or curve.times[-1]>record['cutoff']:raise ValueError('Future or pre-dense prefix')
            sample=snapshot(curve,record['cutoff'])
            if sample is None:raise ValueError('Invalid frozen TRAIN prefix')
            row=cases.loc[record['case_id']].to_dict()
            if (row['event']!=curve.event or row['rank']!=curve.rank or row['start_ms']!=curve.start
                or row['end_ms']!=curve.end or row['cutoff']!=record['cutoff']
                or row['observed_at']!=sample['now'] or not np.isclose(row['current'],sample['score'])):raise ValueError('TRAIN prefix/index mismatch')
            row.update(progress=float(sample['progress']),phase=int(np.searchsorted([.25,.9],sample['progress'],side='right')),split='train')
            if {'actual','target','log_error'}&set(row):raise ValueError('Input index contains outcomes')
            rows.append(row);features.append(feature_row(sample));seen.add(record['case_id'])
    if seen!=set(cases.case_id):raise ValueError('TRAIN prefix panel differs')
    return pd.DataFrame(rows),pd.DataFrame(features)

def execute_c1(run,receipt):
    run=Path(run);cfg=authorization(run,receipt,'C1')
    if (run/'predictions').exists():raise FileExistsError('Preserve interrupted C1 run')
    (run/'predictions').mkdir();inputs=Path(cfg['inputs_directory'])
    records=load_c1_records(inputs);queries=json.loads((inputs/'train_queries.json').read_text())
    ordered=sorted(queries,key=lambda q:(q['start_ms'],q['event']))
    if len(ordered)!=35 or records.event.nunique()!=35 or len(records)!=11536:raise ValueError('Train export changed')
    panels=[];supports=[];costs=[]
    for position,query in enumerate(ordered):
        history,target=release_query(records,event=query['event'],start_ms=query['start_ms'])
        wall=time.monotonic();cpu=time.process_time()
        values,details=c1_predict(history,target,query_start_ms=query['start_ms'])
        period='warmup' if position<15 else 'selection_description' if position<25 else 'temporal_diagnostic'
        emitted=[]
        for method in C1_METHODS:
            frame=target.copy();frame['method']=method;frame['predicted']=values[method]
            frame['effective_bias']=np.log(frame.base_prediction/frame.predicted)
            frame['query_position']=position+1;frame['period']=period
            check_panel(frame,target.case_id);emitted.append(frame)
            for item in details.get(method,[]):supports.append({'event':query['event'],'query_position':position+1,'method':method,**item})
        output=pd.concat(emitted,ignore_index=True)
        prediction_file=run/'predictions'/f'query_{position+1:02d}_event_{query["event"]}.csv.gz'
        output.to_csv(prediction_file,index=False)
        seal_query(run,prediction_file,history,target,position=position+1)
        panels.append(output)
        costs.append({'event':query['event'],'query_position':position+1,'history_events':int(history.event.nunique()),
                      'history_cases':len(history),'target_cases':len(target),'methods':len(values),
                      'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu,'fits':0})
        print(json.dumps(costs[-1]),flush=True)
    pd.concat(panels,ignore_index=True).to_csv(run/'predictions.csv.gz',index=False)
    pd.DataFrame(supports).to_csv(run/'supports.csv',index=False);pd.DataFrame(costs).to_csv(run/'costs.csv',index=False)
    if cfg['source_hashes']!=source_hashes():raise ValueError('C1 code changed during replay')
    return seal(run,'blind_prediction_seal.json')

def _fit_network(x,y,w,rank,qx,qrank,*,structure,epochs,seed):
    """Physical fitting function, called only after the PI receipt gate."""
    import torch
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=network(x.shape[1],structure)
        xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
        wt=torch.tensor(w,dtype=torch.float32);rt=torch.tensor(rank,dtype=torch.long)
        qt=torch.tensor(qx,dtype=torch.float32);qrt=torch.tensor(qrank,dtype=torch.long)
        checkpoints=[];costs=[]
        modules=[('both',model,np.ones(len(x),bool))] if structure=='shared' else [(str(r),model.models[code],rank==code) for code,r in enumerate([1000,2000])]
        for rank_name,module,mask in modules:
            if not mask.any():raise ValueError('Independent rank has no supervision')
            optimizer=torch.optim.AdamW(module.parameters(),lr=.002,weight_decay=.05)
            losses=[];wall=time.monotonic();cpu=time.process_time()
            for _ in range(epochs):
                module.train();optimizer.zero_grad(set_to_none=True)
                predicted=model(xt,rt) if structure=='shared' else module(xt[mask]).squeeze(1)
                weights=wt[mask]/wt[mask].sum()
                loss=(weights*(predicted-yt[mask]).square()).sum()
                if not torch.isfinite(loss):raise ValueError('Nonfinite fit loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            params=sum(p.numel() for p in module.parameters())
            checkpoints.append({'rank':rank_name,'optimizer_state_dict':optimizer.state_dict(),'losses':losses})
            costs.append({'rank':rank_name,'rows':int(mask.sum()),'parameters':params,'epochs':epochs,
                          'optimizer_steps':epochs,'parameter_updates':epochs*params,
                          'wall_seconds':time.monotonic()-wall,'cpu_seconds':time.process_time()-cpu})
        model.eval()
        with torch.no_grad():prediction=model(qt,qrt).numpy().astype(float)
        return prediction,{'state_dict':model.state_dict(),'fit_states':checkpoints,'seed':seed,
                           'features':x.shape[1],'structure':structure,'epochs':epochs},costs

def execute_c2(run,receipt):
    import joblib
    import torch
    from analysis.multirank_model import make_preprocessor
    run=Path(run);cfg=authorization(run,receipt,'C2')
    if (run/'models').exists():raise FileExistsError('Preserve partial/completed C2 run')
    (run/'models').mkdir();dataset=Path(cfg['dataset_directory']);cases,features=load_train_prefixes(dataset)
    clock=cases[['event','start_ms','end_ms']].drop_duplicates().set_index('event')
    qpanels=[];predictions=[];costs=[];group_rows=[];fit_count=0
    rank=cases['rank'].map({1000:0,2000:1}).to_numpy(int)
    for block in cfg['groups']:
        qmask=cases.event.isin(block['query_events']).to_numpy();query=cases.loc[qmask].copy()
        query['endpoint']=block['endpoint'];qrank=rank[qmask];qpanels.append(query)
        for group,events in block['groups'].items():
            mask=cases.event.isin(events).to_numpy();train=cases.loc[mask].copy()
            if (train.end_ms>=block['query_start_ms']).any():raise ValueError('Future TRAIN supervision')
            labels=allowed_train_labels(dataset/'train_labels.csv',train.case_id)
            actual=np.array([labels[k] for k in train.case_id]);y=np.log(actual/train.current.to_numpy())+np.log(train.progress.to_numpy())
            weights=event_rank_weights(train.event.to_numpy(),train['rank'].to_numpy())
            path=run/'models'/f'endpoint{block["endpoint"]}_{group}';path.mkdir()
            prep=make_preprocessor(features.loc[mask]);matrix=prep.fit_transform(features.loc[mask]);qmatrix=prep.transform(features.loc[qmask])
            mean=np.average(matrix,axis=0,weights=weights);std=np.maximum(np.sqrt(np.average((matrix-mean)**2,axis=0,weights=weights)),1e-4)
            nx=np.clip((matrix-mean)/std,-8,8);nqx=np.clip((qmatrix-mean)/std,-8,8)
            joblib.dump({'preprocessor':prep,'mean':mean,'std':std},path/'preprocessing.joblib',compress=3)
            train.assign(weight=weights).to_csv(path/'training_weights.csv',index=False)
            bounds=clock.loc[events];group_rows.append({'endpoint':block['endpoint'],'group':group,'events':len(events),
              'event_ids':json.dumps(events),'rows':len(train),'start_min':int(bounds.start_ms.min()),
              'start_max':int(bounds.start_ms.max()),'end_max':int(bounds.end_ms.max()),
              'span_days':float((bounds.start_ms.max()-bounds.start_ms.min())/86400000),
              'feature_dimension':nx.shape[1],'query_events':json.dumps(block['query_events']),
              'query_cases_sha256':hashlib.sha256(json.dumps(sorted(query.case_id)).encode()).hexdigest()})
            for structure,epochs,name in [('shared',60,'shared60'),('shared',30,'shared30'),('independent',60,'independent60')]:
                members=[]
                for seed in SEEDS:
                    raw,checkpoint,physical_costs=_fit_network(nx,y,weights,rank[mask],nqx,qrank,structure=structure,epochs=epochs,seed=seed)
                    fit_count+=len(physical_costs)
                    if fit_count>cfg['physical_fit_cap']:raise ValueError('Predeclared fit budget exceeded')
                    filename=path/f'{name}_seed{seed}.pt';torch.save(checkpoint,filename)
                    restored=network(nx.shape[1],structure);restored.load_state_dict(torch.load(filename,map_location='cpu',weights_only=True)['state_dict']);restored.eval()
                    with torch.no_grad():replayed=restored(torch.tensor(nqx,dtype=torch.float32),torch.tensor(qrank,dtype=torch.long)).numpy()
                    if not np.allclose(raw,replayed,rtol=1e-6,atol=1e-6):raise ValueError('Saved network replay mismatch')
                    budget=compute_budget(nx.shape[1],len(nx),epochs,structure,rank_rows=[int((rank[mask]==r).sum()) for r in [0,1]])
                    for row in physical_costs:
                        width=32 if structure=='shared' else 16;heads=64 if structure=='shared' else 16
                        costs.append({'endpoint':block['endpoint'],'group':group,'structure':name,'seed':seed,
                                      'forward_linear_MACs':row['rows']*epochs*(width*nx.shape[1]+heads),**row})
                    write_json(path/f'{name}_seed{seed}_budget.json',budget)
                    predicted=query.current.to_numpy()*np.exp(np.clip(raw-np.log(query.progress.to_numpy()),0,4));members.append(predicted)
                    output=query.copy();output['group']=group;output['structure']=name;output['seed']=str(seed)
                    output['method']=f'{group}_{name}_seed{seed}';output['predicted']=predicted
                    check_panel(output,query.case_id);predictions.append(output)
                output=query.copy();output['group']=group;output['structure']=name;output['seed']='mean'
                output['method']=f'{group}_{name}_mean';output['predicted']=np.mean(members,axis=0)
                check_panel(output,query.case_id);predictions.append(output)
            print(json.dumps(group_rows[-1]|{'physical_fits_completed':fit_count}),flush=True)
    if fit_count!=108:raise ValueError('Actual fit count differs from preregistered 108')
    pd.concat(qpanels,ignore_index=True).to_csv(run/'query_cases.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(run/'predictions.csv.gz',index=False)
    pd.DataFrame(costs).to_csv(run/'fit_costs.csv',index=False);pd.DataFrame(group_rows).to_csv(run/'history_groups.csv',index=False)
    write_json(run/'fit_budget.json',{'physical_fits':fit_count,'cap':cfg['physical_fit_cap'],
      'optimizer_steps':int(sum(c['optimizer_steps'] for c in costs)),
      'parameter_updates':int(sum(c['parameter_updates'] for c in costs)),
      'forward_linear_MACs':int(sum(c['forward_linear_MACs'] for c in costs)),
      'fit_cpu_seconds':sum(c['cpu_seconds'] for c in costs),'fit_wall_seconds':sum(c['wall_seconds'] for c in costs)})
    if cfg['source_hashes']!=source_hashes():raise ValueError('C2 code changed during fitting')
    return seal(run,'blind_prediction_seal.json')

def score(run,output,experiment):
    """Independent TRAIN-only scoring process; output is new and separately sealed."""
    run=Path(run);output=Path(output);verify(run,'blind_prediction_seal.json')
    if output.exists():raise FileExistsError('Preserve scored results')
    cfg=json.loads((run/'configuration.json').read_text())
    if cfg['experiment']!=experiment:raise ValueError('Wrong scoring experiment')
    verify_data_hashes(cfg)
    forecasts=pd.read_csv(run/'predictions.csv.gz')
    if experiment=='C1':
        answers=pd.read_csv(Path(cfg['inputs_directory'])/'train_residuals.csv.gz',usecols=['case_id','actual'])
        expected=answers.case_id.tolist();keys=['period','method']
        for _,part in forecasts.groupby('method'):check_panel(part,expected)
    else:
        answers=pd.read_csv(Path(cfg['dataset_directory'])/'train_labels.csv');keys=['endpoint','method']
        query=pd.read_csv(run/'query_cases.csv')
        for (endpoint,_),part in forecasts.groupby(['endpoint','method']):check_panel(part,query.loc[query.endpoint==endpoint].case_id)
    joined=forecasts.merge(answers,on='case_id',how='left',validate='many_to_one')
    output.mkdir(parents=True);summaries=[];by_event=[];detail=[]
    panels={'T1000_T2000':joined[joined['rank'].isin([1000,2000])]} if experiment=='C2' else {'global19':joined,'T1000_T2000':joined[joined['rank'].isin([1000,2000])]}
    for panel,data in panels.items():
        for key,part in data.groupby(keys,sort=False):
            meta=dict(zip(keys,key));summaries.append({'panel':panel,**meta,**metric_summary(part)})
            for event,event_part in part.groupby('event'):by_event.append({'panel':panel,**meta,'event':int(event),**metric_summary(event_part)})
            for dimension in ['rank','phase','event_type','band_group']:
                if dimension not in part:continue
                for value,sub in part.groupby(dimension,dropna=False):detail.append({'panel':panel,**meta,'dimension':dimension,'value':str(value),**metric_summary(sub)})
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(by_event).to_csv(output/'by_event.csv',index=False);pd.DataFrame(detail).to_csv(output/'by_dimension.csv',index=False)
    joined['corrected_log_error']=np.log(joined.predicted/joined.actual)
    aggregations=dict(corrected_mean=('corrected_log_error','mean'),corrected_sd=('corrected_log_error','std'),
                      cases=('case_id','size'),start_ms=('start_ms','first'))
    if experiment=='C1':
        joined['raw_log_error']=np.log(joined.base_prediction/joined.actual)
        aggregations.update(raw_mean=('raw_log_error','mean'),raw_sd=('raw_log_error','std'))
    residual=joined.groupby(keys+['event','rank','phase'],as_index=False).agg(**aggregations)
    residual.to_csv(output/'event_rank_phase_residuals.csv',index=False)
    stability=[]
    for key,part in residual.groupby(keys+['rank','phase'],sort=False):
        for kind in ['corrected']+(['raw'] if experiment=='C1' else []):
            values=part.sort_values(['start_ms','event'])[kind+'_mean'].to_numpy(float)
            corr=float(np.corrcoef(values[:-1],values[1:])[0,1]) if len(values)>2 and np.std(values[:-1])>1e-12 and np.std(values[1:])>1e-12 else None
            stability.append(dict(zip(keys+['rank','phase'],key))|{'residual':kind,'events':len(values),
              'mean_residual':float(values.mean()),'between_event_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
              'mean_absolute_adjacent_change':float(np.abs(np.diff(values)).mean()) if len(values)>1 else None,'lag1_correlation':corr})
    pd.DataFrame(stability).to_csv(output/'cross_event_stability.csv',index=False)
    write_json(output/'scoring_manifest.json',{'experiment':experiment,'blind_seal_sha256':sha(run/'blind_prediction_seal.json'),
      'scoring_source_sha256':sha(__file__),'answers':'TRAIN only, opened in independent scoring process after blind seal',
      'limitations':json.loads((HERE/'preregistration.json').read_text())['limitations']})
    return seal(output,'evaluation_seal.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare-c1','prepare-c2','execute-c1','execute-c2','score-c1','score-c2'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='prepare-c1':result=prepare_c1(a.run)
    elif a.action=='prepare-c2':result=prepare_c2(a.run)
    elif a.action.startswith('execute'):
        if a.authorization is None:p.error('execute requires --authorization PI receipt')
        result=execute_c1(a.run,a.authorization) if a.action=='execute-c1' else execute_c2(a.run,a.authorization)
    else:
        if a.output is None:p.error('score requires new --output directory')
        result=score(a.run,a.output,'C1' if a.action=='score-c1' else 'C2')
    print(json.dumps({'action':a.action,'files':len(result.get('files',{}))}),flush=True)

if __name__=='__main__':main()
