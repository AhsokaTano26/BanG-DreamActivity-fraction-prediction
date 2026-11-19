"""One fixed posthoc transfer check, explicitly after simulation exposure."""
from pathlib import Path
import argparse
import json
import resource
import time
import numpy as np
import pandas as pd
import mechanism as M
import run_mechanism as C

HERE=Path(__file__).resolve().parent


def issue_fixed(history,blind_current,q,r,mu):
    if set(blind_current)-set(C.BLIND) or {'log_error','actual'}&set(blind_current):raise ValueError('Current frame must be blind')
    if blind_current.event.nunique()!=1 or blind_current.start_ms.nunique()!=1:raise ValueError('One blind event required')
    if not blind_current.phase.eq(1).all() or not blind_current['rank'].isin([1000,2000]).all():raise ValueError('Only fixed phase1 pair')
    if (history.end_ms>=blind_current.start_ms.iloc[0]).any() or set(history.event)&set(blind_current.event):raise ValueError('History must end strictly before current start')
    ids=history.sort_values('ordinal').event.drop_duplicates().tolist()
    y,_,events=C.pair_panel(history,ids,1)
    ordinal_map=history.drop_duplicates('event').set_index('event').ordinal
    ordinals=np.array([int(ordinal_map[e]) for e in events]);mean=np.array(mu,float);p=np.eye(2)*.1;previous=-1
    for value,ordinal in zip(y,ordinals):
        prior=p+(ordinal-previous)*q;s=prior+r;gain=np.linalg.solve(s,prior.T).T
        mean+=gain@(value-mean);rem=np.eye(2)-gain
        p=rem@prior@rem.T+gain@r@gain.T;previous=ordinal
    bias=np.clip(mean,-.3,.3);out=blind_current[['case_id','event','rank','phase']].copy()
    out['bias']=[bias[0 if rank==1000 else 1] for rank in out['rank']]
    out['prediction']=np.clip(blind_current.base_prediction.to_numpy()*np.exp(-out.bias),blind_current.current.to_numpy(),blind_current.current.to_numpy()*np.exp(4))
    out['history_complete_pairs']=len(y);out['history_activity_span']=int(ordinals[-1]-ordinals[0]+1) if len(y) else 0
    out['history_max_activity_gap']=int(np.diff(ordinals).max()) if len(y)>1 else 0
    return out


def run(output):
    start=time.process_time();C.source_check()
    seal=json.loads((HERE/'posthoc_transfer_seal.json').read_text())
    for path,digest in seal['sha256'].items():
        if C.sha(HERE/path)!=digest:raise ValueError('Transfer source/input changed: '+path)
    plan=json.loads((HERE/'posthoc_transfer_plan.json').read_text());cal=json.loads((HERE/plan['calibration_seal']).read_text())
    q,r=M.covariances(cal['best']);mu=np.array(cal['mu']);output=Path(output);output.mkdir(exist_ok=False)
    prepared=C.prepare_frame(pd.read_csv(C.INPUTS/'train_residuals.csv.gz'),json.loads((C.INPUTS/'train_queries.json').read_text()))
    frame=prepared['frame'];means=prepared['means'];issued=[];scores=[]
    reference=pd.read_csv(HERE/'calibration_run_v2_001/next_event_signal_benefit.csv')
    reference=reference[(reference.phase==1)&reference.method.isin(['I','S','risk_recent_k8'])]
    for query_index,query in enumerate(prepared['queries'][25:],26):
        current=frame[(frame.event==query['event'])&(frame.phase==1)&frame['rank'].isin([1000,2000])].copy()
        past=means[(means.phase==1)&(means.end_ms<query['start_ms'])&(means.ordinal<query_index-1)].copy()
        prediction=issue_fixed(past,C.blind_frame(current),q,r,mu);issued.append(prediction)
        joined=current.merge(prediction,on=['case_id','event','rank','phase'],validate='one_to_one')
        ape=100*np.abs(joined.prediction/joined.actual-1);loge=(joined.log_error-joined.bias)**2
        comparisons=reference[reference.event==query['event']]
        row={'event':query['event'],'query_index':query_index,'method':'fixed_DLM_posthoc','cases':len(joined),
             'case_mape':float(ape.mean()),'log_mse':float(loge.mean()),'history_complete_pairs':int(prediction.history_complete_pairs.iloc[0]),
             'history_activity_span':int(prediction.history_activity_span.iloc[0]),'history_max_activity_gap':int(prediction.history_max_activity_gap.iloc[0])}
        for ref in comparisons.itertuples(index=False):
            if ref.cases!=len(joined):raise ValueError('Transfer reference case panel mismatch')
            row['MAPE_minus_'+ref.method]=row['case_mape']-ref.case_mape;row['log_mse_minus_'+ref.method]=row['log_mse']-ref.log_mse
        scores.append(row)
    pd.concat(issued).to_csv(output/'issued_DLM.csv',index=False);C.write_csv(output/'event_comparisons.csv',scores)
    summary=pd.DataFrame(scores).drop(columns=['event','query_index','method']).mean().to_dict()
    C.write_json(output/'summary.json',{'identity':'posthoc after simulation and existing train diagnoses; one fixed transfer, no selection',
                 'events':len(scores),'parameters':cal['best'],'calibration_mu':mu,'event_macro':summary})
    C.write_json(output/'run_manifest.json',{'cpu_seconds':time.process_time()-start,'transfer_seal_sha256':C.sha(HERE/'posthoc_transfer_seal.json'),
                 'calibration_seal_sha256':C.sha(HERE/plan['calibration_seal']),
                 'outputs_sha256':{p.name:C.sha(p) for p in output.iterdir() if p.is_file()}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--pi-authorized',action='store_true');args=parser.parse_args()
    if not args.pi_authorized:parser.error('PI allocation required')
    resource.setrlimit(resource.RLIMIT_CPU,(2,2));run(args.output)
