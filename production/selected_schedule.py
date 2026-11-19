"""Switch this project's same launchd job to its explicitly selected worker."""
import json,os,plistlib,sys
from pathlib import Path
from analysis.holdout_dataset import sha256
from production import launchd as old
from production.model import verify_pack
from production.runtime import cycle_lock
from production.state import atomic_json

LABEL=old.LABEL

def plist_spec(config,config_path):
    spec=old.plist_spec(config,config_path)
    spec['ProgramArguments']=['/usr/bin/nice','-n','19',*spec['ProgramArguments']]
    spec['ProgramArguments'][spec['ProgramArguments'].index('production.cli')]='production.selected_cli'
    spec['EnvironmentVariables'].update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return spec

def validate_existing(path,config,config_path):
    if not Path(path).exists():return
    prior=plistlib.loads(Path(path).read_bytes())
    allowed=[old.plist_spec(config,config_path)['ProgramArguments'],plist_spec(config,config_path)['ProgramArguments']]
    if (prior.get('Label')!=LABEL or prior.get('ProgramArguments') not in allowed
        or prior.get('WorkingDirectory')!=str(old.ROOT)):
        raise FileExistsError('Existing job belongs to a different installation')

def validate_loaded(text,config,config_path,plist_path):
    """Fail closed on the actual loaded job, independently of the disk plist."""
    target=f'gui/{os.getuid()}/{LABEL}';lines=text.splitlines()
    if not lines or lines[0].strip()!=target+' = {':raise FileExistsError('Unknown loaded service identity')
    fields={};arguments=None;reading=False
    for line in lines[1:]:
        if reading:
            if line=='\t}':reading=False
            elif line.startswith('\t\t'):arguments.append(line[2:])
            else:raise FileExistsError('Unrecognized loaded argument format')
        elif line=='\targuments = {':
            if arguments is not None:raise FileExistsError('Ambiguous loaded arguments')
            arguments=[];reading=True
        elif line.startswith('\t') and not line.startswith('\t\t'):
            key,separator,value=line[1:].partition(' = ')
            if key in ['path','program','working directory'] and separator:
                if key in fields:raise FileExistsError('Ambiguous loaded identity field')
                fields[key]=value
    allowed=[old.plist_spec(config,config_path)['ProgramArguments'],plist_spec(config,config_path)['ProgramArguments']]
    if reading or set(fields)!=set(['path','program','working directory']) or arguments not in allowed:
        raise FileExistsError('Loaded service belongs to an unknown installation')
    if (not all(Path(v).is_absolute() for v in fields.values())
        or Path(fields['path']).resolve()!=Path(plist_path).resolve()
        or Path(fields['working directory']).resolve()!=old.ROOT.resolve()
        or Path(fields['program']).resolve()!=Path(arguments[0]).resolve()):
        raise FileExistsError('Loaded service path, program or directory differs')
    return dict(target=target,path=fields['path'],program=fields['program'],arguments=arguments,
                working_directory=fields['working directory'])

def loaded_identity(config,config_path,plist_path):
    probe=old._command(['launchctl','print',f'gui/{os.getuid()}/{LABEL}'],check=False)
    if probe.returncode==0:return validate_loaded(probe.stdout,config,config_path,plist_path)
    if 'could not find service' in probe.stderr.lower():return None
    raise RuntimeError('Cannot verify loaded service identity: '+probe.stderr.strip())

def install(config,config_path):
    if sys.platform!='darwin':raise RuntimeError('launchd is only available on macOS')
    if config.get('forecast_method')!='legacy_pair':raise ValueError('Selected default is not configured')
    root=Path(config['root']);pack=json.loads((root/'models/current.json').read_text())
    if not verify_pack(pack['path'])['production_ready']:raise ValueError('Fixture model cannot be scheduled')
    path=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    with cycle_lock(root) as acquired:
        if not acquired:raise RuntimeError('A prediction cycle is running; retry after it completes')
        validate_existing(path,config,config_path)
        loaded=loaded_identity(config,config_path,path)
        if path.exists():
            backup=root/'scheduler_history'/(sha256(path)+'.plist');backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists():assert backup.read_bytes()==path.read_bytes()
            else:backup.write_bytes(path.read_bytes())
        domain=f'gui/{os.getuid()}';target=domain+'/'+LABEL
        if loaded is not None:old._command(['launchctl','bootout',target])
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.plist.tmp')
        temporary.write_bytes(plistlib.dumps(plist_spec(config,config_path),sort_keys=False));temporary.replace(path)
        old._command(['launchctl','bootstrap',domain,str(path)])
        result={'label':LABEL,'domain':domain,'plist':str(path),'config':str(Path(config_path).resolve()),
            'worker':'production.selected_cli','forecast_method':'legacy_pair','interval_seconds':config['interval_seconds']}
        atomic_json(root/'scheduler.json',result);return result

status=old.status

def uninstall(config,config_path=None):
    root=Path(config['root'])
    with cycle_lock(root) as acquired:
        if not acquired:raise RuntimeError('A prediction cycle is running; retry after it completes')
        receipt=json.loads((root/'scheduler.json').read_text()) if (root/'scheduler.json').exists() else {}
        path=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
        if receipt.get('plist') and Path(receipt['plist']).resolve()!=path.resolve():raise FileExistsError('Unknown scheduler receipt')
        config_path=Path(config_path or receipt.get('config',root/'config.json'))
        validate_existing(path,config,config_path)
        loaded=loaded_identity(config,config_path,path)
        if loaded is not None:old._command(['launchctl','bootout',loaded['target']])
        path.unlink(missing_ok=True)
    return {'status':'removed','models_and_data_preserved':True}
