"""Freeze outside-cohort prefixes and replay already sealed models, without fits."""
from __future__ import annotations
import argparse,gzip,importlib,importlib.metadata,json,platform,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
from analysis.backtest import ROOT,Curve,snapshot
from analysis.holdout_dataset import sha256,write_json,curve_record,from_record,prefix_curve
from analysis.multirank_dataset import identity
from analysis.multirank_model import model_prediction
from analysis.temporal_dataset import TemporalDataset,STAGES,origin_groups
from analysis.temporal_rank_subset import DEFAULT
from analysis.temporal_paired_features import prepare_paired
from analysis.temporal_evaluate import verify_predictions
from analysis.research_search import FAMILIES,combine,sources_for
from download_ranks import normalize_cutoffs


def verify_archive(directory,filename='manifest.json'):
    directory=Path(directory);manifest=json.loads((directory/filename).read_text())
    for n,d in manifest['artifacts'].items():
        if sha256(directory/n)!=d:raise ValueError('Archive changed: '+n)
    return manifest


def build(archive,destination,parent,legacy=None):
    archive,destination=Path(archive),Path(destination)
    if destination.exists():raise FileExistsError(destination)
    verify_archive(archive)
    metadata=json.loads((archive/'metadata.json').read_text())
    blocked={e for split in parent.manifest['splits'].values() for e in split['event_ids']}
    boundary=int(pd.read_csv(parent.root/'partitions.csv').end_ms.max())
    records=metadata['rows']
    as_of=round(datetime.fromisoformat(metadata['as_of_utc']).timestamp()*1000) if 'as_of_utc' in metadata else round(datetime.now(timezone.utc).timestamp()*1000)
    if any(r['event'] in blocked or r['event_info']['start_ms']<=boundary for r in records):
        raise ValueError('New activities overlap or precede the frozen cohort')
    if len({(r['event'],r['rank']) for r in records})!=len(records):raise ValueError('Duplicate activity/rank')
    destination.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.forward-',dir=destination.parent))
    try:
        inputs={'grid':[],'legacy':[]};indexes={'grid':[],'legacy':[]};members=[];unavailable=[];outcomes=[];curves={}
        def add(c,cutoff,case,group,stages):
            sample=snapshot(c,cutoff)
            if sample is None:return False
            rec=curve_record(prefix_curve(c,cutoff))|{'case_id':case,'cutoff':cutoff,'stage':stages[0]['stage_id']}
            inputs[group].append(rec)
            indexes[group].append({'case_id':case,'event':c.event,'rank':c.rank,'event_type':c.kind,'band_group':'unknown',
                'cutoff':cutoff,'observed_at':sample['now'],'current':sample['score'],
                'start_ms':c.start,'end_ms':c.end,'observation_age_minutes':(cutoff-sample['now'])/60000})
            if group=='grid':members.extend({'case_id':case,**s} for s in stages)
            return True
        for row in records:
            path=archive/row['raw_file']
            if sha256(path)!=row['raw_sha256']:raise ValueError('Raw observations changed')
            points,_=normalize_cutoffs(json.loads(path.read_text()));info=row['event_info']
            c=Curve(row['event'],row['rank'],info['event_type'],info['start_ms'],info['end_ms'],
                np.array([p['time'] for p in points]),np.array([p['ep'] for p in points],dtype=float))
            if c.rank not in [1000,2000] or c.duration<=0 or len(c.times)<4 or np.any(np.diff(c.times)<=0) or not np.isfinite(c.scores).all() or np.any(c.scores<0):raise ValueError('Invalid new curve')
            curves[c.event,c.rank]=c
            near=[p for p in points if c.end-60000<=p['time']<=c.end+300000]
            final=near[-1] if near else None
            outcomes.append({'event':c.event,'rank':c.rank,'status':'available_near_end' if final else 'missing_near_end',
                'actual':final['ep'] if final else None,'observed_at':final['time'] if final else None,'end_ms':c.end})
            for origin in origin_groups(c):
                cutoff=origin['cutoff'];case=f'forward:{c.event}:{c.rank}:{cutoff}'
                if cutoff>as_of or not c.start<cutoff<c.end or not add(c,cutoff,case,'grid',origin['stages']):
                    unavailable.extend({'event':c.event,'rank':c.rank,'stage_id':s['stage_id'],'cutoff':cutoff,'reason':'insufficient/stale/nonpositive prefix or outside event'} for s in origin['stages'])
        if legacy is not None:
            legacy=Path(legacy);verify_archive(legacy,'prediction_manifest.json')
            prediction=json.loads((legacy/'prediction.json').read_text())
            event=prediction['event_id'];cutoff=round(datetime.fromisoformat(prediction['as_of']).timestamp()*1000)
            old_start=round(datetime.fromisoformat(prediction['event_start']).timestamp()*1000)
            old_end=round(datetime.fromisoformat(prediction['event_end']).timestamp()*1000)
            if not old_start<cutoff<old_end:raise ValueError('Legacy forecast was outside its sealed event schedule')
            old_rows={r['rank']:r for r in prediction['predictions']}
            for rank in [1000,2000]:
                if (event,rank) not in curves:raise ValueError('Legacy activity not in the external cohort')
                points,_=normalize_cutoffs(json.loads((legacy/f'rank_{rank}.json').read_text()))
                c=Curve(event,rank,prediction['event_type'],old_start,old_end,np.array([p['time'] for p in points]),np.array([p['ep'] for p in points],dtype=float))
                sample=snapshot(c,cutoff);expected_time=round(datetime.fromisoformat(old_rows[rank]['observed_at']).timestamp()*1000)
                if sample is None or sample['now']!=expected_time or sample['score']!=old_rows[rank]['current_ep']:raise ValueError('Legacy input differs from sealed prediction state')
                if not add(c,cutoff,f'legacy:{event}:{rank}:{cutoff}','legacy',[{'stage_id':'legacy_snapshot'}]):raise ValueError('Legacy input unavailable')
            shutil.copyfile(legacy/'prediction.json',work/'legacy_forecast.json')
            shutil.copyfile(legacy/'prediction_manifest.json',work/'legacy_source_manifest.json')
            if (legacy/'fetch_manifest.json').exists():shutil.copyfile(legacy/'fetch_manifest.json',work/'legacy_fetch_manifest.json')
        for group in inputs:
            with gzip.open(work/f'{group}_inputs.jsonl.gz','wt') as handle:
                for rec in inputs[group]:handle.write(json.dumps(rec)+'\n')
            pd.DataFrame(indexes[group],columns=['case_id','event','rank','event_type','band_group','cutoff','observed_at','current','start_ms','end_ms','observation_age_minutes']).to_csv(work/f'{group}_cases.csv',index=False)
        pd.DataFrame(members).to_csv(work/'memberships.csv',index=False)
        pd.DataFrame(unavailable,columns=['event','rank','stage_id','cutoff','reason']).to_csv(work/'unavailable.csv',index=False)
        pd.DataFrame(outcomes).to_csv(work/'outcomes.csv',index=False)
        shutil.copyfile(archive/'metadata.json',work/'archive_metadata.json')
        shutil.copyfile(__file__,work/'builder_source.py')
        builder_sources=sources_for({'forward':SimpleNamespace(__file__=__file__)})
        for name in builder_sources:
            target=work/'builder_sources'/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        manifest={'mode':'outside_cohort_retrospective_replay','parent_dataset_id':parent.manifest['dataset_id'],
            'events':sorted({r['event'] for r in records}),'ranks':[1000,2000],'stages':STAGES,
            'archive':str(archive.resolve()),'archive_manifest_sha256':sha256(archive/'manifest.json'),
            'cases':{g:len(v) for g,v in indexes.items()},'created_at':datetime.now(timezone.utc).isoformat(),
            'archive_as_of_ms':as_of,'build_environment':{'python':platform.python_version(),
                'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','httpx','scikit-learn','catboost','joblib']},
                'source_hashes':{n:sha256(ROOT/n) for n in builder_sources}},
            'artifacts':{str(p.relative_to(work)):sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}}
        manifest['dataset_id']=identity(manifest);write_json(work/'manifest.json',manifest);work.rename(destination)
        return manifest
    finally:
        if work.exists():shutil.rmtree(work)


class ForwardDataset:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if identity({k:v for k,v in self.manifest.items() if k!='dataset_id'})!=self.manifest['dataset_id']:raise ValueError('Dataset manifest changed')
        verify_archive(self.root)
    def inputs(self,group='grid'):
        if group not in ['grid','legacy']:raise ValueError(group)
        with gzip.open(self.root/f'{group}_inputs.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);c=from_record(r)
                if c.times[-1]>r['cutoff']:raise ValueError('Future prefix observation')
                sample=snapshot(c,r['cutoff'])
                if sample is None:raise ValueError('Invalid blind prefix')
                yield sample|{'case_id':r['case_id'],'stage':r['stage']}
    def cases(self,group='grid'):return pd.read_csv(self.root/f'{group}_cases.csv')
    def memberships(self):return pd.read_csv(self.root/'memberships.csv')
    def outcomes(self,*,allow=False):
        if not allow:raise ValueError('Outcomes restricted to independent scoring')
        return pd.read_csv(self.root/'outcomes.csv')


def predict(data,model_run,single_run,paired_run,output):
    output=Path(output);model_run=Path(model_run);single_run=Path(single_run);paired_run=Path(paired_run)
    if output.exists():raise FileExistsError(output)
    model_identities={}
    for role,run in [('selected',model_run),('single',single_run),('paired',paired_run)]:
        prior_seal=verify_predictions(run)
        cfg=json.loads((run/'run_manifest.json').read_text())
        if cfg['dataset_id']!=data.manifest['parent_dataset_id'] or prior_seal['dataset_id']!=cfg['dataset_id']:raise ValueError('Frozen model parent dataset differs')
        model_identities[role]={'directory':str(run.resolve()),'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
                                'blind_manifest_sha256':sha256(run/'blind_manifest.json')}
        for name,digest in cfg['source_hashes'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
        if platform.python_version()!=cfg['python'] or any(importlib.metadata.version(n)!=v for n,v in cfg['software'].items()):raise ValueError('Model environment changed')
    config=json.loads((model_run/'run_manifest.json').read_text());selection=json.loads((model_run/'selection.json').read_text())
    chosen=selection['selected_recipe'];spec=selection['combination_specs'][chosen]
    if spec['kind'] not in ['mean','median'] or any(m not in config['recipes'] for m in spec['members']):raise ValueError('Unsupported frozen combination')
    modules={f:importlib.import_module(path) for f,path in FAMILIES.items()}
    sources=sources_for(modules)+['training/analysis/forward_validation.py']
    output.mkdir(parents=True);(output/'models').mkdir();(output/'sources').mkdir()
    manifest={'dataset_id':data.manifest['dataset_id'],'dataset_directory':str(data.root.resolve()),'mode':data.manifest['mode'],
        'model_run':str(model_run.resolve()),'model_seal_sha256':sha256(model_run/'blind_manifest.json'),
        'model_identities':model_identities,
        'selection':chosen,'combination':spec,'created_at':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'software':config['software'],'source_hashes':{n:sha256(ROOT/n) for n in sources}}
    write_json(output/'run_manifest.json',manifest)
    for name in sources:
        target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    bundles={}
    for name in spec['members']:
        shutil.copyfile(model_run/f'models/{name}.joblib',output/f'models/{name}.joblib')
        bundles[name]=joblib.load(output/f'models/{name}.joblib')
    for name,source in [('reference_single',single_run/'models/catboost_rank.joblib'),('reference_paired',paired_run/'models/catboost_paired.joblib')]:
        shutil.copyfile(source,output/f'models/{name}.joblib');bundles[name]=joblib.load(output/f'models/{name}.joblib')
    for group in ['grid','legacy']:
        samples=list(data.inputs(group))
        if not samples:continue
        arrays=prepare_paired(samples)
        predictions={name:modules[config['recipes'][name]['family']].predict_recipe(bundles[name],arrays) for name in spec['members']}
        values={'reference_single':model_prediction(bundles['reference_single'],{'rank':arrays['single'],'current':arrays['current'],'progress':arrays['progress']}),
                'reference_paired':model_prediction(bundles['reference_paired'],arrays),'selected_combo':combine(spec,predictions,arrays)}
        rows=[]
        for method,pred in values.items():
            if not np.isfinite(pred).all() or (pred<=0).any():raise ValueError('Invalid new activity prediction')
            rows.extend({'case_id':c['case_id'],'method':method,'predicted':float(p)} for c,p in zip(arrays['cases'],pred))
        pd.DataFrame(rows).to_csv(output/f'{group}_predictions.csv',index=False)
        arrays['audit'].to_csv(output/f'{group}_pair_audit.csv',index=False)
    seal={k:manifest[k] for k in ['dataset_id','dataset_directory','mode']};seal['methods']=['reference_single','reference_paired','selected_combo']
    seal['artifacts']={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'blind_manifest.json',seal)
    return seal


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    b=commands.add_parser('build');b.add_argument('--archive',type=Path,required=True);b.add_argument('--destination',type=Path,required=True);b.add_argument('--legacy',type=Path)
    f=commands.add_parser('predict');f.add_argument('--dataset',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--model',type=Path,default=ROOT/'analysis/temporal_results/v2_multiagent_20261112')
    f.add_argument('--single',type=Path,default=ROOT/'analysis/temporal_results/v2_t1000_t2000_full_20261112')
    f.add_argument('--paired',type=Path,default=ROOT/'analysis/temporal_results/v2_paired_tcn_20261112')
    a=p.parse_args()
    if a.command=='build':print(build(a.archive,a.destination,TemporalDataset(DEFAULT),a.legacy))
    else:predict(ForwardDataset(a.dataset),a.model,a.single,a.paired,a.output)
