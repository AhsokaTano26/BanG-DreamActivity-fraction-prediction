"""Create reviewable, versioned local delivery archives; no network publishing."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[4]
RECORD=ROOT/'analysis/paper_upgrade_round2/20261118'
PAPER=ROOT/'papers/20261118_entertainment_computing_cn'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def tree(path):
    for p in Path(path).rglob('*'):
        rel=p.relative_to(path)
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name=='.DS_Store':continue
        if path==PAPER and rel.parts[0]=='output' and p.suffix not in ('.pdf','.log'):continue
        if path==RECORD and ('delivery' in rel.parts or (p.suffix=='.zip' and 'baseline' not in rel.parts)):continue
        yield p

def archive(destination,files,description):
    files=sorted(set(files));members={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'description':description,'member_files':len(files),
              'members':members,'manifest_self_hash':'excluded by definition; external delivery receipt hashes the entire ZIP'}
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.writestr('DELIVERY_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(destination) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure '+bad)
    return {'file':str(destination.relative_to(ROOT)),'sha256':sha(destination),'bytes':destination.stat().st_size,
            'members':len(files)+1,'crc':'passed','scope':description}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);a=parser.parse_args()
    if not a.version.replace('_','').isalnum():parser.error('Simple unique version required')
    output=PAPER/'output';output.mkdir(exist_ok=True)
    records=RECORD/'delivery'/a.version;records.mkdir(parents=True,exist_ok=False)
    # The complete original minimum C import closure was independently tested by H.
    closure=['analysis/backtest.py','analysis/event_features.py','analysis/holdout_dataset.py',
      'analysis/holdout_evaluate.py','analysis/multirank_dataset.py','analysis/multirank_model.py',
      'analysis/paper_upgrade_replay_models.py','analysis/rolling_residual_stats.py','download_ranks.py']
    roots=[RECORD,ROOT/'data/temporal_holdout_v2_dense_t1000_t2000_20261115']
    files=list(tree(PAPER))+[ROOT/p for p in closure]+[ROOT/'AGENTS.md']
    for p in roots:files+=list(tree(p))
    old=ROOT/'analysis/paper_upgrade_records/20261118'
    for name in ['reports','synthetic']:
        files+=list(tree(old/name))
    for name in ['fixed_replay_v2','learning_curve_v2','matched_train_v3','interval_joint_and_learning_contrasts_v1']:
        files+=list(tree(old/'results/statistics'/name))
    files += [old/'runs/two_rank_learning_curve_v2/fit_costs.csv']
    for p in (old/'prospective').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files.append(p)
    files += list(tree(old/'portable_v1/capsule_v2'))
    files += [ROOT/'analysis/paper_upgrade_prospective.py',ROOT/'analysis/v3_next_calibration.py',
              ROOT/'analysis/data_quality/20261115_sampling_density/summary.json']
    for split in ['test','validation']:
        for kind in ['intervals','predictions']:
            # Base point predictions are already retained compactly in the portable capsule.
            if kind=='intervals':files.append(old/'runs/fixed_replay_v2'/f'{split}_{kind}.csv')
    for name in ['test_cases.csv','test_labels.csv','manifest.json']:
        files.append(ROOT/'data/rolling_holdout_v3_dense_19rank_20261117'/name)
    for folder in ['specs','plans']:
        for pattern in ['2026-11-18-paper*','2026-11-15-dense-cohorts*']:
            files+=list((ROOT/'docs/superpowers'/folder).glob(pattern))
    # Include every declared numerical/figure source, beyond the explicit closure.
    for name in ['upgrade_number_bindings.json','round2_number_bindings.json']:
        binding=json.loads((PAPER/'qa'/name).read_text())
        for rel in binding['sources']:
            source=ROOT/rel
            if not source.is_file():raise FileNotFoundError('Bound source absent: '+str(source))
            files.append(source)
    figures=json.loads((PAPER/'qa/upgrade_figure_manifest.json').read_text())
    for figure in figures['figures']:
        for item in figure['inputs']:
            source=ROOT/item['path']
            if not source.is_file():raise FileNotFoundError('Figure source absent: '+str(source))
            files.append(source)
    receipts=[]
    native=list(PAPER.glob('*.tex'))+list(PAPER.glob('*.bib'))+[PAPER/'Makefile',PAPER/'README.md',PAPER/'requirements-upgrade.txt',PAPER/'delivery_manifest.json',PAPER/'delivery_receipt.json']
    native+=list((PAPER/'figures').glob('upgrade_*.pdf'))+list((PAPER/'tables').glob('*.tex'))+list((PAPER/'data').glob('*macros.tex'))
    native+=list((PAPER/'scripts').glob('*.py'))
    for name in ['upgrade_number_bindings.json','round2_number_bindings.json','upgrade_figure_manifest.json','round2_document_checks.json','round2_reference_audit.csv','font_check.json']:
        native.append(PAPER/'qa'/name)
    receipts.append(archive(output/f'round2_latex_source_{a.version}.zip',native,
       'Native Chinese LaTeX main/supplement and all18 vector PDF figures; editable SVG/PNG and numerical regeneration evidence are in the separate figure/research bundles.'))
    receipts.append(archive(output/f'round2_research_bundle_{a.version}.zip',files,
       'Second-round code, actual fitting checkpoints, train-only inputs, independent audits, complete numerical bindings and paper; 19-rank data contains a scoring subset, full original base-model retraining caches remain in the original workspace.'))
    receipts.append(archive(output/f'round2_figures_{a.version}.zip',list(tree(RECORD/'workspaces/F')),
       '18 vector PDF/editable SVG/400dpi PNG figure groups, plotting code, provenance and visual QA.'))
    receipt={'status':'packaged_crc_verified','version':a.version,'archives':receipts}
    (records/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
