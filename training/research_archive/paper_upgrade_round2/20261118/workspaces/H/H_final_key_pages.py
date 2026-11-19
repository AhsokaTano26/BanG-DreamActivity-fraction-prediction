"""PI-assigned five-second independent rendering of six canonical PDF pages."""
import hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2];PAPER=ROOT/'papers/20261118_entertainment_computing_cn';G=BASE/'workspaces/G/manuscript';OUT=HERE/'H_final_key_pages';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();results=[];errors=[]
for kind,page in [('main',16),('main',18),('main',21),('main',42),('supplement',18),('supplement',19)]:
    left=5-(time.monotonic()-start)
    if left<=0:errors.append('five-second budget reached');break
    output=OUT/f'{kind}{page}';command=['pdftoppm','-f',str(page),'-l',str(page),'-singlefile','-r','96','-png',str(PAPER/'output'/f'{kind}.pdf'),str(output)]
    try:
        process=subprocess.run(command,capture_output=True,text=True,timeout=left,check=True)
        target=output.with_suffix('.png');reference=G/'qa/round2_render'/kind/f'page-{page:02d}.png';results.append({'kind':kind,'page':page,'source_pdf_sha256':sha(PAPER/'output'/f'{kind}.pdf'),'PNG':str(target.relative_to(HERE)),'PNG_sha256':sha(target),'G_reference_sha256':sha(reference),'same_bytes_as_G':sha(target)==sha(reference)})
    except Exception as exc:errors.append({'kind':kind,'page':page,'error':str(exc)});break
elapsed=time.monotonic()-start;out={'scope':'Fresh96dpi rendering directly from canonical root PDFs; no model runs','budget_seconds':5,'elapsed_seconds':elapsed,'pages':results,'errors':errors,'passed':len(results)==6 and all(r['same_bytes_as_G'] for r in results) and not errors,'script_sha256':sha(__file__)}
(HERE/'H_final_key_page_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
