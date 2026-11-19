"""Independent sealed scoring for19rank V3 and explicitly separate panels."""
from __future__ import annotations
import argparse,importlib.metadata,json,platform,shutil,tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from analysis.backtest import ROOT
from analysis.holdout_dataset import sha256
from analysis.innovation_common import local_sources
from analysis.multirank_dataset import identity
from analysis.multiline_dataset import MultilineDataset
from analysis.multiline_runner import RUN
from analysis.rolling_benchmark import verify_seal,write_new_json
from analysis.temporal_evaluate import aligned_frame,measured,score_temporal
from analysis.rolling_residual_evaluate import intervals

def panel_metrics(cases,labels,predictions,*,ranks=None):
    expected=set(cases.case_id)
    if cases.case_id.duplicated().any() or predictions.case_id.duplicated().any() or set(predictions.case_id)!=expected or set(labels)!=expected:raise ValueError('Exact frozen full panel required before subset')
    subset=cases if ranks is None else cases[cases['rank'].isin(ranks)]
    if subset.empty:return measured(pd.DataFrame())
    ids=set(subset.case_id)
    return measured(aligned_frame(subset,{c:labels[c] for c in ids},predictions[predictions.case_id.isin(ids)]))

def evaluate(run,output=None):
    run=Path(run);output=Path(output or run);blind=verify_seal(run/'blind_manifest.json')
    if blind['protocol']!='rolling_multiline_calibration_v3':raise ValueError('Expected multiline calibration run')
    data=MultilineDataset(blind['dataset_directory'])
    if data.manifest['dataset_id']!=blind['dataset_id']:raise ValueError('Dataset identity differs')
    sources=local_sources([Path(__file__)])
    ident={'blind_manifest_sha256':sha256(run/'blind_manifest.json'),'dataset_id':blind['dataset_id'],
        'source_hashes':{n:sha256(ROOT/n) for n in sources},'python':platform.python_version(),
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','torch','joblib']},
        'draws':5000,'seed':42,'blocks':[3,5],'reference':'ewma_reference','panels':['all19','original2','cold1500','shared18']}
    eid=identity(ident);receipt=output/'evaluation_manifest.json'
    if receipt.exists():
        old=verify_seal(receipt)
        if old['evaluation_id']!=eid:raise FileExistsError('Scorer changed; use new --output')
        return json.loads((output/'evaluation.json').read_text())
    names=['evaluation.json','metrics.csv','by_rank.csv','by_stage.csv','by_stage_rank.csv','by_type.csv','by_band.csv','panel_metrics.csv','paired_comparisons.json','rank_support.csv','evaluator_sources']
    if any((output/n).exists() for n in names):raise FileExistsError('Preserve partial/previous scoring')
    selection=json.loads((run/'selection.json').read_text());cfg=json.loads((run/'run_manifest.json').read_text())
    if selection['selected_recipe']!=blind['selected_recipe'] or set(selection['methods'])!=set(blind['methods']):raise ValueError('Selection binding differs')
    result={'evaluation_id':eid,'run_id':blind['run_id'],'dataset_id':blind['dataset_id'],'selection':selection,
        'exposure':cfg['exposure'],'metadata_caveat':cfg['metadata_caveat'],'training_scope':'32basewarmup +35prequential calibration activities; not resubstitution',
        'reference':'independent rankwise warm EWMA using the same35forecast residuals','splits':{},'panels':{},'paired_comparisons':[],'panel_comparisons':{}}
    panels={'all19':cfg['ranks'],'original2':[1000,2000],'cold1500':[1500],'shared18':[r for r in cfg['ranks'] if r!=1500]}
    metrics=[];rankrows=[];stagerows=[];crossrows=[];typerows=[];bandrows=[];panelrows=[]
    for split in ['validation','test']:
        cases=data.case_index(split);members=data.memberships(split);labels=data.labels(split,allow_test=split=='test')
        predictions=pd.read_csv(run/f'{split}_predictions.csv')
        if set(predictions.method)!=set(blind['methods']):raise ValueError('Missing method panel')
        result['splits'][split]={};result['panels'][split]={p:{} for p in panels}
        for method,p in predictions.groupby('method'):
            value=score_temporal(cases,members,labels,p,stage_catalog=data.manifest['protocol']['stages']);result['splits'][split][method]=value
            tags={'split':split,'method':method,'selected':method==selection['selected_recipe']};metrics.append(tags|value['overall'])
            rankrows.extend(tags|{'rank':r}|v for r,v in value['by_rank'].items())
            stagerows.extend(tags|{'stage':s}|v for s,v in value['by_stage'].items());crossrows.extend(tags|v for v in value['stage_rank'])
            typerows.extend(tags|{'event_type':t}|v for t,v in value.get('by_event_type',{}).items());bandrows.extend(tags|{'band_group':b}|v for b,v in value.get('by_band_group',{}).items())
            for panel,ranks in panels.items():
                measured_panel=panel_metrics(cases,labels,p,ranks=ranks);result['panels'][split][panel][method]=measured_panel
                panelrows.append(tags|{'panel':panel}|measured_panel)
        if split=='validation':
            for method,value in selection['validation_scores'].items():
                if not np.isclose(result['splits'][split][method]['overall']['event_macro_mape_pct'],value,rtol=0,atol=1e-9):raise ValueError('Frozen validation scores differ')
        else:
            for panel,ranks in panels.items():
                subset=cases[cases['rank'].isin(ranks)];ids=set(subset.case_id)
                order=sorted(set(subset.event),key=lambda e:(data.event_index[int(e)]['start_ms'],int(e)))
                comparison=intervals(subset,{c:labels[c] for c in ids},predictions[predictions.case_id.isin(ids)],order) if order else []
                result['panel_comparisons'][panel]=comparison
            result['paired_comparisons']=result['panel_comparisons']['all19']
    support=pd.read_csv(data.root/'partitions.csv').groupby(['rank','split']).size().unstack(fill_value=0).reset_index()
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.multiline-evaluation-',dir=output.parent))
    try:
        write_new_json(stage/'evaluation.json',result);write_new_json(stage/'paired_comparisons.json',{'draws':5000,'seed':42,'reference':'ewma_reference','panels':result['panel_comparisons']})
        for name,rows in [('metrics',metrics),('by_rank',rankrows),('by_stage',stagerows),('by_stage_rank',crossrows),('by_type',typerows),('by_band',bandrows),('panel_metrics',panelrows)]:pd.DataFrame(rows).to_csv(stage/f'{name}.csv',index=False)
        support.to_csv(stage/'rank_support.csv',index=False)
        for name in sources:
            p=stage/'evaluator_sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        manifest={'evaluation_id':eid,'identity':ident,'artifacts':{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
        manifest['seal_id']=identity(manifest);write_new_json(stage/'evaluation_manifest.json',manifest)
        output.mkdir(parents=True,exist_ok=True)
        for name in names+['evaluation_manifest.json']:
            if (output/name).exists():raise FileExistsError('Scoring destination appeared')
            (stage/name).rename(output/name)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(pd.DataFrame(metrics).query("split=='test'")[['method','selected','event_macro_mape_pct']].to_string(index=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--output',type=Path);a=p.parse_args();evaluate(a.run,a.output)
