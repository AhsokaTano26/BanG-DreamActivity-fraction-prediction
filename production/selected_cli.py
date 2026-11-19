"""Current operational entrypoint for the user-selected best tested recipe."""
import argparse,json
from pathlib import Path
from production.selected_runtime import DEFAULT_ROOT,read_config,run_cycle
from production import selected_schedule
from production.state import Ledger

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    for name in ['cycle','status','install','uninstall']:
        s=commands.add_parser(name);s.add_argument('--config',type=Path,default=DEFAULT_ROOT/'config.json')
        if name=='cycle':s.add_argument('--offline',action='store_true')
    a=p.parse_args(argv)
    try:
        cfg=read_config(a.config);root=Path(cfg['root'])
        if a.command=='cycle':result=run_cycle(cfg,sync=not a.offline)
        elif a.command=='install':result=selected_schedule.install(cfg,a.config)
        elif a.command=='uninstall':result=selected_schedule.uninstall(cfg,a.config)
        else:
            result={'config':cfg,'ledger':Ledger(root).summary(),'scheduler':selected_schedule.status(cfg)}
            if (root/'status.json').exists():result['last_cycle']=json.loads((root/'status.json').read_text())
            if (root/'models/current.json').exists():result['prepared_model']=json.loads((root/'models/current.json').read_text())
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0
    except KeyboardInterrupt:return 0
    except Exception as ex:
        print(json.dumps({'status':'error','error':f'{type(ex).__name__}: {ex}'},ensure_ascii=False));return 1

if __name__=='__main__':raise SystemExit(main())
