"""Validate the current round-two PDFs and binding manifest; no numerical models run."""
from pathlib import Path
import argparse,json,re,subprocess,hashlib,csv,math
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper-root',required=True,type=Path);p.add_argument('--repo-root',required=True,type=Path);p.add_argument('--bindings-only',action='store_true');a=p.parse_args();P=a.paper_root.resolve();ROOT=a.repo_root.resolve();Q=P/'qa';Q.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd):return subprocess.run(cmd,capture_output=True,text=True,check=True).stdout
results={};errors=[]
for kind in ([] if a.bindings_only else ['main','supplement']):
 pdf=P/'output'/f'{kind}.pdf';log=(P/'output'/f'{kind}.log').read_text(errors='replace')
 info=run(['pdfinfo',str(pdf)]);fonts=run(['pdffonts',str(pdf)]);text=run(['pdftotext','-layout',str(pdf),'-'])
 warnings=[line for line in log.splitlines() if any(k in line for k in ['Overfull \\hbox','Overfull \\vbox','undefined','Missing character:','multiply defined'])]
 unembedded=[line for line in fonts.splitlines()[2:] if re.search(r'\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line)]
 if warnings:errors.append((kind,'LaTeX warnings',warnings))
 if unembedded:errors.append((kind,'unembedded fonts',unembedded))
 if '??' in text:errors.append((kind,'unresolved text references'))
 if '0811.0405Author' in text:errors.append((kind,'preprint note spacing'))
 results[kind]={'pdf_sha256':sha(pdf),'pages':int(re.search(r'Pages:\s+(\d+)',info).group(1)),'fonts':fonts,'latex_warnings':warnings,'characters':len(text)}
 (Q/f'round2_{kind}_pdffonts.txt').write_text(fonts);(Q/f'round2_{kind}_text.txt').write_text(text)
# Every referenced figure must exist after final F integration.
files=['main.tex','supplement.tex','round2_studies.tex','round2_supplement.tex']
figs=[]
for f in files:
 s=(P/f).read_text()
 figs+=re.findall(r'\\(?:upfigure|roundfigure)\{([^}]+)\}',s)
figs=[name for name in figs if not name.startswith('#')]
for name in ([] if a.bindings_only else figs):
 if not (P/'figures'/f'{name}.pdf').is_file():errors.append(('missing figure',name))
binding_checks={}
for manifest in ['upgrade_number_bindings.json','round2_number_bindings.json']:
 j=json.loads((Q/manifest).read_text())
 for rel,digest in j['generated_files'].items():
  if sha(P/rel)!=digest:errors.append(('generated table mismatch',rel))
 for rel,digest in j['sources'].items():
  if not (ROOT/rel).is_file() or sha(ROOT/rel)!=digest:errors.append(('source hash mismatch',rel))
 macrofile=P/'data'/('round2_macros.tex' if manifest.startswith('round2') else 'upgrade_macros.tex')
 if sha(macrofile)!=j['macros_sha256']:errors.append(('macro hash mismatch',str(macrofile)))
 if not manifest.startswith('round2'):
  binding_checks[manifest]={'sources':len(j['sources']),'bindings':len(j['bindings']),'scope':'legacy source/table/macro hashes; original numerical bindings retained'}
  continue
 macros=dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]+)\}',macrofile.read_text()))
 checked=0
 for b in j['bindings']:
  path=ROOT/b['source'];field=b['field'];where=b['row_condition']
  if sha(path)!=b['source_sha256']:errors.append(('binding source hash',b['binding']));continue
  if path.suffix=='.csv':
   with path.open(newline='') as h:rr=list(csv.DictReader(h))
   rr=[r for r in rr if all(str(r.get(k))==str(v) for k,v in where.items())]
  else:
   obj=json.loads(path.read_text())
   if 'object'in where:obj=obj[where['object']]
   rr=[obj]
  try:
   if b.get('aggregation','').startswith('negative/positive'):
    vv=[float(r[field]) for r in rr];raw=[sum(x<0 for x in vv),sum(x>0 for x in vv)]
    if raw!=b['value'] or '/'.join(map(str,raw))!=b['rendered']:raise ValueError('direction count')
   elif b.get('aggregation')=='R_raw_00 / R_raw_11':
    if len(rr)!=1:raise ValueError('derived row identity')
    raw=float(rr[0][field[0]])/float(rr[0][field[1]])
    if raw!=b['value'] or f"{raw:.2f}"!=b['rendered']:raise ValueError('ratio')
   else:
    if len(rr)!=1:raise ValueError('row identity')
    raw=rr[0][field];v=float(raw)*b.get('multiplier',1)
    if str(raw)!=str(b['raw_value']) or v!=b['value']:raise ValueError('raw or transformed value')
    digits=int(re.search(r'(\d+) decimals',b['rounding']).group(1))
    if f'{v:.{digits}f}'!=b['rendered']:raise ValueError('rendered rounding')
   if b['binding'] in macros and macros[b['binding']]!=b['rendered']:raise ValueError('macro field')
   if not b.get('unit'):raise ValueError('missing unit')
   checked+=1
  except (ValueError,KeyError,IndexError,TypeError) as e:errors.append(('binding arithmetic',b['binding'],str(e)))
 binding_checks[manifest]={'sources':len(j['sources']),'bindings':len(j['bindings']),'independently_checked_fields':checked,'macros':len(macros)}
results['bindings']=binding_checks
results['figure_count']=len(figs);results['unique_figures']=len(set(figs));results['errors']=errors
results['status']='passed' if not errors else 'failed'
(Q/('round2_binding_checks.json' if a.bindings_only else 'round2_document_checks.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pages':{k:v['pages'] for k,v in results.items() if isinstance(v,dict) and 'pages'in v},'figures':len(set(figs)),'status':results['status'],'errors':errors},ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
