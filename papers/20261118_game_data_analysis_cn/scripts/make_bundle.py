"""Package the paper and verify a source-only build outside the working tree."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

P=Path(__file__).resolve().parents[1]


def main():
    rootnames=['main.tex','numbers.tex','references.bib','Makefile','README.md','highlights.txt','requirements-analysis.txt']
    files=[P/name for name in rootnames]
    for folder in ['tables','figures','data','scripts']:
        files.extend(p for p in (P/folder).iterdir() if p.is_file() and p.name!='cleaned_qualified_curves.jsonl.gz')
    files.extend([P/'output/中文论文.pdf'])
    files.extend(p for p in (P/'qa').iterdir() if p.is_file() and
                 ((p.suffix in ['.json','.txt','.log'] and p.name not in ['standalone_bundle_validation.json','standalone_build.log','bundle.log']) or p.name.endswith('_gray.png')))
    files=sorted(set(files))
    manifest={'paper':'BanG Dream国服活动排名分数的时序结构与活动条件关联','author':'Tano',
              'source_rows_sha256':json.loads((P/'data/source_manifest.json').read_text())['canonical_selected_rows_sha256'],
              'scope':'Second descriptive paper; complete current source snapshot and derivations. Legacy cache supplied as terminal score excerpt and audit.',
              'files':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (P/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    files.append(P/'delivery_manifest.json')
    archive=P/'game_data_analysis_cn_source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(Path(P.name)/p.relative_to(P)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(str(Path(P.name)/name))).hexdigest()==digest,name
        with tempfile.TemporaryDirectory(prefix='bangdream_data_paper_') as temp:
            z.extractall(temp);work=Path(temp)/P.name
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            latex=shutil.which('xelatex') or '/Library/TeX/texbin/xelatex'
            bibtex=shutil.which('bibtex') or '/Library/TeX/texbin/bibtex'
            with (P/'qa/standalone_build.log').open('w') as log:
                subprocess.run(['make','pdf','LATEX='+latex,'BIBTEX='+bibtex],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'-B','scripts/validate_paper.py'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'-B','scripts/verify_artifacts.py'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            texttool=shutil.which('pdftotext') or '/opt/homebrew/bin/pdftotext'
            original=subprocess.check_output([texttool,'-layout',str(P/'output/中文论文.pdf'),'-'])
            rebuilt=subprocess.check_output([texttool,'-layout',str(work/'output/中文论文.pdf'),'-'])
            assert original==rebuilt,'Standalone manuscript text differs'
            assert not any(s in (work/'output/main.log').read_text() for s in ['Warning','Overfull','Underfull','Missing character'])
    result={'status':'passed','archive':archive.name,'entries':len(files),'bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc_verified':True,'all_entry_hashes_verified':True,
            'standalone_latex_build':True,'standalone_independent_numerical_checks':True,'standalone_artifact_checks':True,
            'standalone_extracted_text_identical':True}
    (P/'qa/standalone_bundle_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
