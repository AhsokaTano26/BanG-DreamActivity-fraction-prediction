"""Sealed sequential V3 shared/rank/peer ablations, anchored to issued EWMA."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.rolling_benchmark import verify_seal,seal_directory,write_new_json
from analysis.multiline_dataset import MultilineDataset,DEFAULT as DATA
from analysis.multiline_runner import INPUT_COLUMNS
from analysis.task_peer_data import PeerCache,DEFAULT as PEERS
from analysis import task_learning_models as models

RUN=ROOT/'analysis/rolling_results/v3_task_learning_20261118'
REFERENCE=ROOT/'analysis/rolling_results/v3_multiline_calibration_20261117'
SPEC=ROOT/'docs/superpowers/specs/2026-11-18-task-learning-design.md'
REFERENCE_ID='14888ceeff945c1fd387a0e03265ddde1283901a654a678b9b81ac4a71699f5d'
PEER_CACHE_SEAL='015fe62951413c8237bcbda98814c07d131135acb1e3bb56458cff3b3591fdf2'
CONTROLS={'uncorrected':'uncorrected','ewma_reference':'ewma_reference','legacy_pair':'legacy_pair',
          'previous_neural':'neural_gate','previous_selected_graph':'graph1'}
CANDIDATES=['ewma_reference',*models.MODES]
METHODS=[*CONTROLS,*models.MODES,*[f'{m}_seed{s}' for m in models.MODES for s in models.SEEDS]]


def validate_peer_binding(data,cache,*,allow_fixture):
    sealed=verify_seal(cache.root/'manifest.json')
    if sealed!=cache.manifest or sealed['dataset_id']!=data.manifest['dataset_id']:raise ValueError('Peer cache header/dataset changed')
    if cache.ranks!=data.manifest['protocol']['ranks'] or sealed['ranks']!=cache.ranks:raise ValueError('Peer rank scope differs')
    if allow_fixture:return
    if sealed['seal_id']!=PEER_CACHE_SEAL:raise ValueError('Formal peer cache was not preregistered')
    expected=['manifest.json',*[f'{s}_{kind}' for s in ['train','validation','test'] for kind in ['inputs.jsonl.gz','cases.csv']]]
    if set(sealed['source_identity'])!=set(expected):raise ValueError('Peer source set differs')
    for name in expected:
        if sealed['source_identity'][name]!=sha256(data.root/name):raise ValueError('Peer blind source differs: '+name)
    if sealed['artifacts']['source.py']!=sha256(ROOT/'analysis/task_peer_data.py'):raise ValueError('Peer builder source differs')


def configuration(data,reference,cache,allow_fixture):
    validate_peer_binding(data,cache,allow_fixture=allow_fixture)
    seal=verify_seal(reference/'blind_manifest.json');old=json.loads((reference/'run_manifest.json').read_text())
    if old['run_id']!=seal['run_id'] or identity({k:v for k,v in old.items() if k!='run_id'})!=old['run_id']:raise ValueError('Reference run identity differs')
    if data.manifest['dataset_id']!=seal['dataset_id'] or cache.manifest['dataset_id']!=seal['dataset_id']:raise ValueError('Dataset/cache/reference differ')
    if not allow_fixture and (old['run_id']!=REFERENCE_ID or old['allow_fixture']):raise ValueError('Wrong formal reference')
    for name,digest in old['source_hashes'].items():
        if sha256(ROOT/name)!=digest or sha256(reference/'sources'/name)!=digest:raise ValueError('Historical source changed')
    for split in ['train','validation','test']:
        if set(cache.cases[cache.cases.split==split].case_id)!=set(data.case_index(split).case_id):raise ValueError('Peer cache lost frozen cases')
    sources=local_sources([Path(__file__)])
    cfg={'protocol':'rolling_multiline_calibration_v3','experiment':'small_shared_rank_peer_residuals',
         'dataset_id':seal['dataset_id'],'dataset_directory':str(data.root.resolve()),'ranks':list(cache.ranks),
         'base_run':str(reference.resolve()),'reference_run_id':old['run_id'],'reference_blind_sha256':sha256(reference/'blind_manifest.json'),
         'peer_cache':str(cache.root.resolve()),'peer_cache_manifest_sha256':sha256(cache.root/'manifest.json'),
         'warmup_events':old['warmup_events'],'warm_forecast_events':old['warm_forecast_events'],
         'methods':METHODS,'candidates':CANDIDATES,'descriptive_controls':CONTROLS,'reference_method':'ewma_reference',
         'recipes':models.RECIPE,'epochs':2 if allow_fixture else 60,'minimum_events':2 if allow_fixture else 12,
         'selection':'minimum validation event-macro over anchor and4new seed ensembles; descriptive old controls and seeds ineligible',
         'history':'only own sealed earlier forecast events strictly ended before current start; per-row graphs retained from original issuance',
         'training_scores':'35prequential activities after32basewarmup; not fitted errors',
         'source_hashes':{n:sha256(ROOT/n) for n in sources},'specification_sha256':sha256(SPEC),
         'python':platform.python_version(),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','torch','catboost','joblib','scikit-learn']},
         'resource_policy':{'processes':1,'threads':1,'nice':19,'background':True},'allow_fixture':bool(allow_fixture),
         'exposure':'Previously explored test, reused validation; heldout blind-feature statistics also explored',
         'metadata_caveat':data.manifest['protocol']['metadata_caveat']}
    cfg['run_id']=identity(cfg);return cfg


def run(data,run,*,reference_run=REFERENCE,peer_cache=None,allow_fixture=False,resume=False):
    run,reference=Path(run),Path(reference_run);cache=peer_cache or PeerCache()
    cfg=configuration(data,reference,cache,allow_fixture)
    if run.exists():
        if not resume:raise FileExistsError('New immutable run directory required')
        if json.loads((run/'run_manifest.json').read_text())!=cfg or not (run/'blind_manifest.json').exists():raise ValueError('Only identical complete run can resume')
        verify_seal(run/'blind_manifest.json');return json.loads((run/'selection.json').read_text())
    run.mkdir(parents=True);(run/'events').mkdir();write_new_json(run/'run_manifest.json',cfg)
    for name in cfg['source_hashes']:
        p=run/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    queries=sorted([r for r in data.event_index.values() if r['event'] in cfg['warm_forecast_events'] or r['split']!='train'],key=lambda r:(r['start_ms'],r['event']))
    cases={s:data.case_index(s) for s in ['train','validation','test']}
    history=None;released={};states={};predictions={}

    def release_before(start):
        nonlocal history
        for row in sorted(queries,key=lambda r:(r['end_ms'],r['start_ms'],r['event'])):
            event=row['event']
            if row['end_ms']>=start or event in released:continue
            path=run/'events'/str(event);sealed=verify_seal(path/'event_manifest.json')
            if sealed['run_id']!=cfg['run_id'] or sealed['event']!=event:raise ValueError('Own feedback seal mismatch')
            finals={c.rank:c.final for c in data.reveal_event(event,before_ms=start)}
            f=pd.read_csv(path/'features.csv',float_precision='round_trip')
            f['reference_log_error']=[float(np.log(r.reference_prediction/finals[int(r.rank)])) for r in f.itertuples()]
            history=f if history is None or history.empty else pd.concat([history,f],ignore_index=True)
            states[event]=json.loads((path/'relation_state.json').read_text())
            released[event]={k:row[k] for k in ['event','split','start_ms','end_ms','sha256']}
            released[event]['prediction_seal_sha256']=sha256(path/'event_manifest.json')

    def predict_event(row):
        nonlocal history
        started=time.monotonic();event,start=row['event'],row['start_ms'];release_before(start)
        oldpath=reference/'events'/str(event);oldseal=verify_seal(oldpath/'event_manifest.json')
        oldaudit=json.loads((oldpath/'audit.json').read_text())
        if (oldseal['run_id']!=cfg['reference_run_id'] or oldseal['event']!=event or oldseal['query_start_ms']!=start
            or oldseal['end_ms']!=row['end_ms'] or oldaudit['current_outcome_released'] is not False):raise ValueError('Reference query clock mismatch')
        expected={r['event'] for r in queries if r['end_ms']<start}
        if {r['event'] for r in oldaudit['released_forecasts']}!=expected:raise ValueError('Reference forecast history cohort differs')
        raw=pd.read_csv(oldpath/'features.csv',float_precision='round_trip')
        frame=raw[[*INPUT_COLUMNS,'base_prediction']].copy()
        source=pd.read_csv(oldpath/'predictions.csv',float_precision='round_trip');control={}
        for new,old in CONTROLS.items():
            p=source[source.method==old]
            if p.case_id.duplicated().any() or set(p.case_id)!=set(frame.case_id):raise ValueError('Missing reference case/method')
            control[new]=p.set_index('case_id').loc[frame.case_id].predicted.to_numpy(float)
        frame['reference_prediction']=control['ewma_reference']
        wanted=cases[row['split']].query('event==@event').set_index('case_id')
        if frame.case_id.duplicated().any() or set(frame.case_id)!=set(wanted.index):raise ValueError('Frozen current cases changed')
        target=cache.batch(frame.case_id)
        for a in frame.itertuples():
            c=wanted.loc[a.case_id];b=target.frame.set_index('case_id').loc[a.case_id]
            if (int(a.rank)!=int(c['rank']) or a.start_ms!=c.start_ms or a.end_ms!=c.end_ms or a.cutoff!=c.cutoff
                or a.observed_at!=b.observed_at or not np.isclose(a.current,c.current,rtol=1e-12,atol=1e-6)):
                raise ValueError('Reference own prefix disagrees with blind case/cache')
        if history is None:
            history=frame.iloc[:0].copy();history['reference_log_error']=pd.Series(dtype=float)
        graph=models.relation_state(history,query_start_ms=start,ranks=cfg['ranks']);states[event]=graph
        directory=Path(tempfile.mkdtemp(prefix=f'.event-{event}-',dir=run/'events'))
        write_new_json(directory/'relation_state.json',graph)
        past=cache.batch(history.case_id)
        learned,details=models.fit_predict(past,target,history.copy(deep=True),frame,states,query_start_ms=start,
                 model_directory=directory/'models',epochs=cfg['epochs'],minimum_events=cfg['minimum_events'])
        scores=control|learned
        if set(scores)!=set(METHODS):raise ValueError('Method menu differs')
        rows=[]
        for name,values in scores.items():
            if len(values)!=len(frame) or not np.isfinite(values).all() or (values<frame.current.to_numpy()).any():raise ValueError('Invalid or missing prediction')
            for c,v in zip(frame.to_dict('records'),values):
                rows.append({k:c[k] for k in ['case_id','event','rank','phase','current']}|{'method':name,'predicted':float(v),'log_bias':float(np.log(c['base_prediction']/v))})
        pd.DataFrame(rows).to_csv(directory/'predictions.csv',index=False);frame.to_csv(directory/'features.csv',index=False)
        history.to_csv(directory/'released_residuals.csv',index=False)
        write_new_json(directory/'audit.json',{'event':event,'split':row['split'],'query_start_ms':start,'current_outcome_released':False,
            'released_forecasts':list(released.values()),'reference_event_manifest_sha256':sha256(oldpath/'event_manifest.json'),
            'reference_history':oldaudit['released_forecasts'],'peer_cache_manifest_sha256':cfg['peer_cache_manifest_sha256'],
            'selection_sha256':sha256(run/'selection.json') if row['split']=='test' else None,
            'training_rows':len(history),'history_events':int(history.event.nunique()),'network_status':details['status']})
        seal_directory(directory,'event_manifest.json',{'run_id':cfg['run_id'],'event':event,'split':row['split'],'query_start_ms':start,
            'end_ms':row['end_ms'],'cases':len(frame),'methods':METHODS})
        directory.rename(run/'events'/str(event))
        print(json.dumps({'sealed_event':event,'split':row['split'],'history_events':len(released),'cases':len(frame),
                          'network_status':details['status'],'seconds':round(time.monotonic()-started,2)}),flush=True)
        return pd.DataFrame(rows)

    for split in ['train','validation','test']:
        events=[r for r in queries if r['split']==split]
        if split=='test':
            release_before(events[0]['start_ms']);truth=history.set_index('case_id');scores={}
            for name in CANDIDATES:
                p=predictions['validation'].query('method==@name');h=truth.loc[p.case_id]
                actual=h.reference_prediction.to_numpy()*np.exp(-h.reference_log_error.to_numpy())
                errors=pd.DataFrame({'event':p.event.to_numpy(),'ape':np.abs(p.predicted.to_numpy()/actual-1)*100})
                scores[name]=float(errors.groupby('event').ape.mean().mean())
            selected=min(CANDIDATES,key=lambda k:(scores[k],CANDIDATES.index(k)))
            write_new_json(run/'selection.json',{'run_id':cfg['run_id'],'selected_recipe':selected,'candidates':CANDIDATES,'methods':METHODS,
                 'validation_scores':scores,'criterion':cfg['selection'],'test_labels_used_for_selection':False,
                 'sealed_before_test_start_ms':events[0]['start_ms'],
                 'validation_event_seals':{str(r['event']):sha256(run/'events'/str(r['event'])/'event_manifest.json') for r in queries if r['split']=='validation'}})
        predictions[split]=pd.concat([predict_event(r) for r in events],ignore_index=True)
        expected=set(cases[split][cases[split].event.isin(cfg['warm_forecast_events'])].case_id) if split=='train' else set(cases[split].case_id)
        for _,p in predictions[split].groupby('method'):
            if p.case_id.duplicated().any() or set(p.case_id)!=expected:raise ValueError('Exact case panel changed')
        predictions[split].to_csv(run/('warm_predictions.csv' if split=='train' else f'{split}_predictions.csv'),index=False)
    for name,digest in cfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Source changed during experiment')
    seal_directory(run,'blind_manifest.json',{'protocol':cfg['protocol'],'run_id':cfg['run_id'],'dataset_id':cfg['dataset_id'],
        'dataset_directory':cfg['dataset_directory'],'methods':METHODS,'selected_recipe':selected,'reference_method':'ewma_reference'})
    return json.loads((run/'selection.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--reference-run',type=Path,default=REFERENCE);p.add_argument('--peer-cache',type=Path,default=PEERS);p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(json.dumps(run(MultilineDataset(a.dataset),a.run,reference_run=a.reference_run,peer_cache=PeerCache(a.peer_cache),resume=a.resume),ensure_ascii=False))
