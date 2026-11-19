"""Train once, predict continuously, and learn completed new activities locally."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from production.runtime import DEFAULT_ROOT,initialize,read_config,run_cycle,rollback,cycle_lock
from production.state import Ledger
from production import launchd


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    init=commands.add_parser('init');init.add_argument('--root',type=Path,default=DEFAULT_ROOT);init.add_argument('--ranks',default=None);init.add_argument('--interval',type=int,default=900)
    for name in ['cycle','run','status','install','uninstall','rollback','resume-auto']:
        sub=commands.add_parser(name);sub.add_argument('--config',type=Path,default=DEFAULT_ROOT/'config.json')
        if name in ['cycle','run']:sub.add_argument('--offline',action='store_true')
        if name=='rollback':sub.add_argument('--model-id',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='init':
            ranks=None if a.ranks is None else [int(v.strip()) for v in a.ranks.split(',')]
            result=initialize(a.root,ranks=ranks,interval_seconds=a.interval)
        else:
            cfg=read_config(a.config);root=Path(cfg['root'])
            if a.command=='cycle':result=run_cycle(cfg,sync=not a.offline)
            elif a.command=='run':
                while True:
                    try:
                        result=run_cycle(cfg,sync=not a.offline)
                        print(json.dumps({'status':result['status'],'as_of_ms':result['as_of_ms'],'model_id':result.get('model_id'),'predictions':len(result.get('predictions',[]))},ensure_ascii=False),flush=True)
                    except Exception as exc:print(json.dumps({'status':'error','error':str(exc)},ensure_ascii=False),flush=True)
                    time.sleep(cfg['interval_seconds'])
            elif a.command=='status':
                result={'config':cfg,'ledger':Ledger(root).summary(),'scheduler':launchd.status(cfg)}
                if (root/'status.json').exists():result['last_cycle']=json.loads((root/'status.json').read_text())
                if (root/'models/current.json').exists():result['prepared_model']=json.loads((root/'models/current.json').read_text())
            elif a.command=='install':result=launchd.install(cfg,a.config)
            elif a.command=='uninstall':result=launchd.uninstall(cfg)
            elif a.command=='rollback':result=rollback(cfg,a.model_id)
            else:
                with cycle_lock(root) as acquired:
                    if not acquired:raise RuntimeError('Cycle is running; retry after completion')
                    (root/'models/manual_override.json').unlink(missing_ok=True)
                result={'status':'automatic_training_and_promotion_resumed'}
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except KeyboardInterrupt:return 0
    except Exception as exc:
        print(json.dumps({'status':'error','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False))
        return 1

if __name__=='__main__':raise SystemExit(main())
