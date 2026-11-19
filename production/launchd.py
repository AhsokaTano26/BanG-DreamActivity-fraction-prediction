"""macOS user schedule for bounded forecasting cycles; no root privileges."""
from __future__ import annotations
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from analysis.backtest import ROOT
from production.state import atomic_json
from production.model import verify_pack

LABEL='com.tano.bangdream-forecast'


def plist_spec(config,config_path):
    root=Path(config['root']).resolve()
    return {'Label':LABEL,'ProgramArguments':[str(ROOT/'.venv/bin/python'),'-m','production.cli','cycle','--config',str(Path(config_path).resolve())],
            'WorkingDirectory':str(ROOT),'StartInterval':int(config['interval_seconds']),'RunAtLoad':True,
            'ProcessType':'Background','StandardOutPath':str(root/'scheduler.stdout.log'),
            'StandardErrorPath':str(root/'scheduler.stderr.log'),
            'EnvironmentVariables':{'DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}}


def validate_existing(path,spec):
    if not Path(path).exists():return
    prior=plistlib.loads(Path(path).read_bytes())
    if prior.get('Label')!=LABEL or prior.get('ProgramArguments')!=spec['ProgramArguments']:
        raise FileExistsError('An existing schedule points to a different installation; preserve it and choose explicitly')


def _command(args,*,check=True):
    result=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if check and result.returncode:raise RuntimeError(' '.join(args)+': '+result.stderr.strip())
    return result


def install(config,config_path):
    if sys.platform!='darwin':raise RuntimeError('Use production.cli run on non-macOS hosts')
    root=Path(config['root']);pointer=root/'models/current.json'
    pack=json.loads(pointer.read_text());manifest=verify_pack(pack['path'])
    if not manifest['production_ready']:raise ValueError('Train a production model before installing automatic service')
    spec=plist_spec(config,config_path);directory=Path.home()/'Library/LaunchAgents';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{LABEL}.plist';validate_existing(path,spec)
    domain=f'gui/{os.getuid()}';target=domain+'/'+LABEL
    if _command(['launchctl','print',target],check=False).returncode==0:
        _command(['launchctl','bootout',target])
    temporary=path.with_suffix('.plist.tmp');temporary.write_bytes(plistlib.dumps(spec,sort_keys=False));temporary.replace(path)
    _command(['launchctl','bootstrap',domain,str(path)])
    result={'label':LABEL,'plist':str(path),'domain':domain,'config':str(Path(config_path).resolve()),'interval_seconds':config['interval_seconds'],
            'operation':'Runs for the logged-in user while the Mac is awake; StartInterval is not a cloud24/7 host.'}
    atomic_json(root/'scheduler.json',result)
    return result


def uninstall(config):
    root=Path(config['root']);receipt=root/'scheduler.json'
    if not receipt.exists():return {'status':'not_installed'}
    spec=json.loads(receipt.read_text());path=Path(spec['plist']);expected=plist_spec(config,Path(spec['config']))
    validate_existing(path,expected)
    _command(['launchctl','bootout',spec['domain']+'/'+LABEL],check=False)
    path.unlink(missing_ok=True)
    receipt.rename(root/'scheduler.removed.json')
    return {'status':'removed','models_and_data_preserved':True}


def status(config):
    root=Path(config['root']);path=root/'scheduler.json'
    if not path.exists():return {'installed':False}
    receipt=json.loads(path.read_text());result=_command(['launchctl','print',receipt['domain']+'/'+LABEL],check=False)
    return {'installed':result.returncode==0,**receipt,'launchctl':result.stdout if result.returncode==0 else result.stderr}
