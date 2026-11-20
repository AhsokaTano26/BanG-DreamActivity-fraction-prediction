"""Selected recipe worker; existing activity pins retain their original policy."""
from __future__ import annotations
import asyncio,csv,io,json,os,tempfile,time
from datetime import datetime
from pathlib import Path
from analysis.backtest import snapshot,TZ
from analysis.holdout_dataset import prefix_curve
from production.data import Store,sync_data
from production.state import Ledger,atomic_json
from production.model import train_pack,verify_pack
from production.selected_model import build_bias,predict_pack
from production.selected_feedback import seed_feedback,ensure_archive
from production.runtime import (DEFAULT_ROOT,read_config,validate_root,cycle_lock,_activate,_previous,_append_log)

def run_cycle(config,*,now_ms=None,sync=True,transport=None):
    if config.get('forecast_method')!='legacy_pair':raise ValueError('Select legacy_pair in runtime config first')
    root=validate_root(config['root']);now=int(time.time()*1000) if now_ms is None else int(now_ms)
    with cycle_lock(root) as acquired:
        if not acquired:return {'status':'busy','as_of_ms':now,'predictions':[]}
        errors=[];start=time.monotonic();store=Store(root);ledger=Ledger(root)
        try:
            ensure_archive(ledger)
            if not ledger.fact('bootstrap_complete'):
                store.bootstrap(database=Path(config['bootstrap_database']),metadata_directory=Path(config['bootstrap_metadata']),now_ms=now)
                ledger.fact('bootstrap_complete',True)
            atomic_json(root/'status.json',{'status':'running','phase':'sync','as_of_ms':now,'ledger':ledger.summary()})
            sync_result={'status':'offline','errors':[]}
            if sync:
                try:sync_result=asyncio.run(sync_data(store,now_ms=now,ranks=config['ranks'],transport=transport))
                except Exception as exc:errors.append({'stage':'sync','error':f'{type(exc).__name__}: {exc}'})
                errors.extend({'stage':'sync','error':str(e)} for e in sync_result.get('errors',[]))
            events=store.events(now)
            active=[e for e in events if e['start_ms']<=now<e['end_ms']]
            before=min((e['start_ms'] for e in active),default=now)
            completed=store.load_curves(now,ranks=config['ranks'],events=ledger.forecast_events())
            feedback_updates=ledger.learn_completed(completed,now_ms=now,grace_ms=config.get('feedback_grace_ms',600000))
            try:seed_result=seed_feedback(store,ledger,config,before_ms=before,now_ms=now)
            except Exception as exc:
                errors.append({'stage':'calibration_seed','error':f'{type(exc).__name__}: {exc}'});seed_result={'status':'failed'}
            override=root/'models/manual_override.json'
            single_model=not config.get('automatic_training',True)
            training=config.get('automatic_training',True) and not override.exists()
            history=store.load_curves(before,ranks=config['ranks']) if training else []
            atomic_json(root/'status.json',{'status':'running','phase':'training' if training else 'model_loading',
                'as_of_ms':now,'eligible_curves':len(history) if training else None,'ledger':ledger.summary()})
            if override.exists():
                chosen=json.loads(override.read_text());model_path=Path(chosen['path']);manifest=verify_pack(model_path)
                if manifest['history_max_end_ms']>=before:raise ValueError('Rollback model is not causal for this event')
            elif single_model:
                model_path,manifest=_previous(root,before_ms=before,current_only=True)
            else:
                try:
                    if config.get('require_history_coverage'):
                        from production.deployment import history_covers_model
                        try:previous_path,previous_manifest=_previous(root,before_ms=before)
                        except ValueError:previous_path=None
                        if previous_path is not None and not history_covers_model(previous_path,history):
                            raise ValueError('Incomplete training history: retain released weights until all previous history pairs are available')
                    model_path=train_pack(history,store.metadata_directory,root/'models/versions',before_ms=before,ranks=config['ranks'])
                    manifest=verify_pack(model_path);_activate(root,model_path,manifest)
                except Exception as exc:
                    errors.append({'stage':'training','error':f'{type(exc).__name__}: {exc}'})
                    model_path,manifest=_previous(root,before_ms=before)
            curve_map={(c.event,c.rank):c for c in store.load_curves(now,ranks=config['ranks'],
                complete_only=False,events=[e['event'] for e in active])}
            output=[];added=0;pinned_models={};pending_pins=[]
            for event in active:
                pin=ledger.pin(event['event'])
                if pin is None:
                    if seed_result.get('status')=='failed':
                        pending_pins.append(event['event'])
                        output.extend({'event':event['event'],'rank':r,'available':False,
                            'reason':'New activity waits for verified calibration seed; retry next cycle'} for r in config['ranks'])
                        continue
                    calibration=build_bias(ledger.feedback_rows(before_ms=event['start_ms']),before_ms=event['start_ms'],exclude_event_id=event['event'])
                    pin=ledger.create_pin(event,model_path,manifest['model_id'],calibration,now_ms=now)
                pinned=Path(pin['model_path'])
                if single_model:
                    if manifest['history_max_end_ms']>=event['start_ms']:
                        raise ValueError('Installed model uses current or future event data')
                    if pin['model_id']!=manifest['model_id'] or pinned.resolve()!=Path(model_path).resolve():
                        pin=ledger.replace_unusable_pin(event['event'],model_path,manifest['model_id'],
                            reason='Installed model replaced previous version',now_ms=now)
                    pinned=Path(model_path);pin_manifest=manifest
                else:
                    try:pin_manifest=verify_pack(pinned)
                    except (ValueError,OSError,KeyError) as exc:
                        replacement,replacement_manifest=_previous(root,before_ms=event['start_ms'])
                        errors.append({'stage':'pin_recovery','event':event['event'],'error':str(exc),'replacement':replacement_manifest['model_id']})
                        pin=ledger.replace_unusable_pin(event['event'],replacement,replacement_manifest['model_id'],reason=str(exc),now_ms=now)
                        pinned=replacement;pin_manifest=replacement_manifest
                if pin_manifest['history_max_end_ms']>=event['start_ms']:raise ValueError('Pinned model uses current or future event data')
                samples=[];unavailable=[]
                for rank in config['ranks']:
                    c=curve_map.get((event['event'],rank))
                    prefix=None if c is None else prefix_curve(c,now)
                    sample=None if prefix is None else snapshot(prefix,now)
                    if sample is None:
                        unavailable.append({'event':event['event'],'rank':rank,'available':False,'reason':'missing, insufficient, or observations older than60minutes'});continue
                    sample.update(case_id=f"live:{event['event']}:{rank}:{now}",stage='live');samples.append(sample)
                rows=predict_pack(pinned,samples,metadata_directory=store.metadata_directory,calibration=pin['calibration']) if samples else []
                added+=ledger.record(event,rows,cutoff_ms=now,issued_ms=now,model_id=pin['model_id'])
                output.extend(rows+unavailable)
                pinned_models[str(event['event'])]={'model_id':pin['model_id'],'path':str(pinned),'forecast_method':pin['calibration'].get('forecast_method','pinned_independent_ewma')}
            result={'status':'degraded' if errors else ('ok' if active else 'waiting_for_event'),'as_of_ms':now,
                'as_of':datetime.fromtimestamp(now/1000,TZ).isoformat(),'ranks':config['ranks'],
                'model_id':manifest['model_id'],'model_path':str(Path(model_path).resolve()),
                'automatic_model_updates':config.get('automatic_training',True) and not override.exists(),
                'active_event_models':pinned_models,'selected_forecast_method':'legacy_pair',
                'pending_activity_pins':pending_pins,
                'active_events':[e['event'] for e in active],'predictions':sorted(output,key=lambda r:(r['event'],r['rank'])),
                'new_issued_forecasts':added,'feedback_updates':feedback_updates,'calibration_seed':seed_result,
                'ledger':ledger.summary(),'sync':sync_result,'errors':errors,'duration_seconds':round(time.monotonic()-start,3),
                'scope':'Production on all eligible completed dense histories; benchmark test metrics do not describe this latest fit.'}
            atomic_json(root/'latest.json',result);atomic_json(root/'status.json',{k:v for k,v in result.items() if k!='predictions'})
            buffer=io.StringIO();fields=list(dict.fromkeys(k for r in result['predictions'] for k in r))
            if fields:
                writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(result['predictions'])
            path=root/'latest.csv';fd,tmp=tempfile.mkstemp(prefix='.latest-',dir=root)
            with os.fdopen(fd,'w') as f:f.write(buffer.getvalue())
            Path(tmp).replace(path)
            _append_log(root,{'as_of_ms':now,'status':result['status'],'model_id':manifest['model_id'],'predictions':len(output),'new_forecasts':added,'feedback_updates':feedback_updates,'errors':errors})
            return result
        except Exception as exc:
            failure={'status':'error','as_of_ms':now,'error':f'{type(exc).__name__}: {exc}','errors':errors,'ledger':ledger.summary()}
            atomic_json(root/'status.json',failure);_append_log(root,failure)
            raise
