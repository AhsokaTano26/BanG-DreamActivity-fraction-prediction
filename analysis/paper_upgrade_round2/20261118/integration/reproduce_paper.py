"""One serial command: bind sealed evidence, redraw figures, compile and verify."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

DEFAULT_ROOT=Path(__file__).resolve().parents[4]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--paper-root',type=Path)
    p.add_argument('--skip-figures',action='store_true',help='Compile existing vector assets without regenerating font-dependent figures.')
    a=p.parse_args();root=a.repo_root.resolve();paper=(a.paper_root or root/'papers/20261118_entertainment_computing_cn').resolve()
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
    if hasattr(os,'nice'):os.nice(max(0,19-os.nice(0)))
    latex=shutil.which('xelatex') or '/Library/TeX/texbin/xelatex'
    bib=shutil.which('bibtex') or '/Library/TeX/texbin/bibtex'
    def run(cmd,cwd=root):
        print(json.dumps({'running':list(map(str,cmd))}),flush=True)
        subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,check=True)
    run([sys.executable,'-B',paper/'scripts/bind_round2_evidence.py','--repo-root',root,'--paper-root',paper])
    if not a.skip_figures:
        run([sys.executable,'-B',root/'analysis/paper_upgrade_round2/20261118/workspaces/F/scripts/make_round2_figures.py',
             '--repo-root',root,'--out-root',paper,'--qa'])
    options=['PYTHON='+sys.executable,'REPO_ROOT='+str(root),'LATEX='+latex,'BIBTEX='+bib]
    run(['make','all',*options],paper);run(['make','verify',*options],paper);run(['make','render',*options],paper)
    render_path=paper/'qa/round2_render_manifest.json'
    rendered=json.loads(render_path.read_text())
    for kind,record in rendered.items():
        record['source_pdf_sha256']=hashlib.sha256((paper/'output'/f'{kind}.pdf').read_bytes()).hexdigest()
        record['renderer_sha256']=hashlib.sha256((paper/'scripts/render_round2.py').read_bytes()).hexdigest()
    render_path.write_text(json.dumps(rendered,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','paper':str(paper/'output/中文论文.pdf'),'supplement':str(paper/'output/补充材料.pdf'),
                      'scope':'Evidence/figures/native PDF reproduction; no new fitting and no future results.'}),flush=True)

if __name__=='__main__':main()
