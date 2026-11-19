"""Independent read-only PDF identity, fonts, text bounds and suite-log audit."""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(a):return subprocess.run(a,capture_output=True,text=True,check=True,timeout=20).stdout
def run(paper):
    checks={};documents={};render=json.loads((paper/'qa/round2_render_manifest.json').read_text());expected={'main':('897ad0e707bd1f57c0911c73fb96b56f348e96c1396624a3368c5c492143acb3',45),'supplement':('455bb5f09fb1a18305510d08b54d71f5231ad5a9a67659229216395e015e066b',19)}
    for kind,(digest,pages_expected) in expected.items():
        p=paper/'output'/f'{kind}.pdf';info=call(['pdfinfo',str(p)]);font=call(['pdffonts',str(p)]);bbox=call(['pdftotext','-bbox',str(p),'-']);text=call(['pdftotext','-layout',str(p),'-']);tree=ET.fromstring(bbox);outside=[];page_words=[]
        for i,page in enumerate([v for v in tree.iter() if v.tag.endswith('page')],1):
            w=float(page.get('width'));h=float(page.get('height'));count=0
            for word in page.iter():
                if not word.tag.endswith('word'):continue
                count+=1
                if float(word.get('xMin'))<-.1 or float(word.get('yMin'))<-.1 or float(word.get('xMax'))>w+.1 or float(word.get('yMax'))>h+.1:outside.append({'page':i,'word':''.join(word.itertext())})
            page_words.append(count)
        unembedded=[row for row in font.splitlines()[2:] if not re.search(r'\s+yes\s+(?:yes|no)\s+yes\s+\d+\s+\d+\s*$',row)];log=(paper/'output'/f'{kind}.log').read_text(errors='replace');warnings=[line for line in log.splitlines() if any(s in line for s in ['Overfull \\hbox','Overfull \\vbox','undefined','Missing character:','multiply defined'])];renderbad=[r for r,h in render[kind]['page_sha256'].items() if sha(paper/r)!=h];pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));refs=text.rsplit('参考文献',1)[-1];reference_ids=[int(v) for v in re.findall(r'^\s*\[(\d+)\]',refs,re.M)]
        documents[kind]={'pdf_sha256':sha(p),'expected_sha256':digest,'pages':pages,'words_per_page':page_words,'words_outside_page':outside,'font_report':font,'unembedded_or_without_unicode':unembedded,'latex_warnings':warnings,'render_manifest_page_files':len(render[kind]['page_sha256']),'render_manifest_changed_files':renderbad,'references':reference_ids,'unresolved_markers':('??' in text or '0811.0405Author' in text),'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        checks[kind+'_identity_fonts_bounds_references']={'passed':sha(p)==digest and pages==pages_expected and not outside and not unembedded and not warnings and not renderbad and len(page_words)==pages_expected and reference_ids==list(range(1,34 if kind=='main' else 4)) and not documents[kind]['unresolved_markers']}
        (HERE/f'H_final_{kind}_text.txt').write_text(text)
    verify=json.loads((BASE/'verification_summary.json').read_text());badlogs=[];observed=[]
    for row in verify['logs']:
        p=ROOT/row['log'];s=p.read_text(errors='replace');match=re.findall(r'Ran\s+(\d+)\s+tests?\s+in',s);number=int(match[-1]) if match else None
        if sha(p)!=row['sha256'] or number!=row['tests'] or not re.search(r'\nOK\s*$',s):badlogs.append(row['scope'])
        observed.append({'scope':row['scope'],'tests':number,'log_sha256':sha(p)})
    checks['canonical539_test_log_evidence']={'passed':not badlogs and verify['unique_test_cases_total']==539 and sum(r['tests'] for r in observed)==539,'detail':{'canonical_summary_sha256':sha(BASE/'verification_summary.json'),'bad_logs':badlogs,'observed':observed,'identity':'PI executed tests; H rechecked logs and source hashes, no new full-suite rerun.'}}
    out={'scope':'Independent PDF parsing and source/log identity. Existing64-page contacts inspected; fresh key-page rendering recorded separately.','checks':checks,'documents':documents,'render_manifest_sha256':sha(paper/'qa/round2_render_manifest.json'),'paper_root':str(paper),'script_sha256':sha(__file__),'status':'passed' if all(v['passed'] for v in checks.values()) else 'review'}
    (HERE/'H_final_document_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':out['status'],'checks':checks,'documents':{k:{'pages':v['pages'],'SHA':v['pdf_sha256'],'bounds_issues':len(v['words_outside_page'])} for k,v in documents.items()}},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper-root',type=Path,default=BASE/'workspaces/G/manuscript');run(p.parse_args().paper_root.resolve())
