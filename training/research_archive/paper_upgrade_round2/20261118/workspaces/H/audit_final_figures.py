"""Read-only frozen F provenance, vector export and derived-profile audit."""
from __future__ import annotations
import csv,hashlib,json,re,subprocess,sys,xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2];F=BASE/'workspaces/F';G=BASE/'workspaces/G/manuscript'
HELPER=Path('/Users/tano/.codex/plugins/cache/research-skills/figures/0.13.2/agents/figure-qa-scripts/check_svg.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def call(argv):return subprocess.run(argv,capture_output=True,text=True,timeout=10)
manifest=j(F/'qa/upgrade_figure_manifest.json');delivery=j(F/'F_delivery_manifest.json');checks={};records=[]
def check(k,v,detail=None):checks[k]={'passed':bool(v),'detail':detail}
bad=[rel for rel,digest in delivery['files'].items() if sha(F/rel)!=digest]
check('F_delivery_all_files_sha',not bad,{'files':len(delivery['files']),'bad':bad,'delivery_manifest_sha256':sha(F/'F_delivery_manifest.json')})
check('F_18_semantic_figures',len(manifest['figures'])==18 and len({r['name'] for r in manifest['figures']})==18)
helper_out=HERE/'H_final_svg_programmatic';helper_out.mkdir(exist_ok=True)
for figure in manifest['figures']:
    name=figure['name'];badinputs=[r['path'] for r in figure['inputs'] if sha(ROOT/r['path'])!=r['sha256']];badformats=[];copybad=[]
    for form,meta in figure['formats'].items():
        if sha(F/meta['path'])!=meta['sha256']:badformats.append(form)
        if sha(G/meta['path'])!=meta['sha256']:copybad.append(form)
    svg=F/figure['formats']['svg']['path'];tree=ET.parse(svg);ns={'s':'http://www.w3.org/2000/svg'};images=tree.findall('.//s:image',ns);texts=tree.findall('.//s:text',ns);paths=tree.findall('.//s:path',ns);ids=[v.get('id') for v in tree.iter() if v.get('id')]
    pdf=F/figure['formats']['pdf']['path'];pi=call(['pdfimages','-list',str(pdf)]);font=call(['pdffonts',str(pdf)]);bbox=call(['pdftotext','-bbox',str(pdf),'-']);doc=ET.fromstring(bbox.stdout);pages=[x for x in doc.iter() if x.tag.endswith('page')];outside=[];words=0
    for page in pages:
        width=float(page.get('width'));height=float(page.get('height'))
        for word in page.iter():
            if not word.tag.endswith('word'):continue
            words+=1
            if float(word.get('xMin'))<-.1 or float(word.get('yMin'))<-.1 or float(word.get('xMax'))>width+.1 or float(word.get('yMax'))>height+.1:outside.append(''.join(word.itertext()))
    pdf_images=[line for line in pi.stdout.splitlines() if re.match(r'^\s*\d+\s+\d+\s+',line)];fontrows=font.stdout.splitlines()[2:];unembedded=[line for line in fontrows if not re.search(r'\s+yes\s+(?:yes|no)\s+yes\s+\d+\s+\d+\s*$',line)]
    result=call([sys.executable,str(HELPER),str(svg),'--journal','generic','--json']);(helper_out/(name+'.json')).write_text(result.stdout);(helper_out/(name+'.stderr')).write_text(result.stderr)
    programmatic=json.loads(result.stdout) if result.stdout.strip() else None
    sizes=[]
    for text in texts:
        for match in re.finditer(r'font-size:\s*([0-9.]+)(px|pt)',text.get('style','')):sizes.append(float(match.group(1)))
    records.append({'name':name,'bad_source_sha':badinputs,'bad_format_sha':badformats,'G_copy_mismatch':copybad,'svg_image_objects':len(images),'svg_live_text_objects':len(texts),'svg_paths':len(paths),'svg_unique_ids':len(ids)==len(set(ids)),'pdf_image_objects':len(pdf_images),'pdf_fonts_unembedded_or_missing_unicode':unembedded,'pdf_words':words,'pdf_text_outside_page':outside,'helper_returncode':result.returncode,'helper_findings':programmatic.get('findings',[]) if programmatic else [],'helper_availability':'unavailable (script error)' if result.returncode==2 else 'available','caption':figure['caption'],'scientific_purpose':figure['scientific_purpose'],'evidence_type':figure['evidence_type'],'minimum_nominal_fontsize_pt':figure['minimum_fontsize_pt'],'minimum_explicit_SVG_fontsize_user_units':min(sizes) if sizes else None})
    if name=='upgrade_calibrated_transfer':
        counts={}
        for group in tree.findall('.//s:g',ns):
            if not group.get('id','').startswith('PathCollection_'):continue
            # Ignore legend copies, which have no data clipping. Some native
            # markers are repeated <use>, whereas triangles are individual paths.
            clipped=[node for node in group.iter() if node.get('clip-path')]
            if not clipped:continue
            points=sum(len(node.findall('.//s:use',ns)) if node.tag.endswith('g') else int(node.tag.endswith('path')) for node in clipped)
            counts[group.get('id')]=points
        check('calibrated_transfer_scatter_cardinality',list(counts.values())==[900,17,24,24,24,24],counts)
check('F_all_sources_and_outputs_and_G_copies',all(not r['bad_source_sha'] and not r['bad_format_sha'] and not r['G_copy_mismatch'] for r in records))
check('F_SVG_PDF_vector_no_images',all(r['svg_image_objects']==r['pdf_image_objects']==0 and r['svg_paths']>0 and r['svg_live_text_objects']>0 and r['svg_unique_ids'] for r in records))
check('F_pdf_fonts_and_page_bounds',all(not r['pdf_fonts_unembedded_or_missing_unicode'] and not r['pdf_text_outside_page'] for r in records))
# Independent scalar grouping; no plotting source imported or executed.
profile=rows(BASE/'workspaces/D/calibration_run_v2_001/calibration_profile.csv');curve=rows(F/'data/figure_data/round2_profile_ratio_curve.csv');groups=defaultdict(list)
for row in profile:groups[round(float(row['q_r_ratio']),12)].append(float(row['two_delta_nll']))
badcurve=[]
for row in curve:
    key=float(row['q_r_ratio_key']);values=groups.get(key,[])
    if not values or abs(min(values)-float(row['two_delta_nll']))>1e-12 or len(values)!=int(row['grid_points']):badcurve.append(row)
support=[r for r in profile if r['support']=='True']
check('profile_round12_min_independent',len(curve)==len(groups) and not badcurve and len(support)==17,{'raw_ratio_unique':len({float(r['q_r_ratio']) for r in profile}),'rounded_ratio_unique':len(groups),'grid_points':len(profile),'support_points':len(support),'bad_rows':badcurve,'derived_sha256':sha(F/'data/figure_data/round2_profile_ratio_curve.csv')})
out={'scope':'Read-only18 frozen figures; no generation or rendering. Independent XML/PDF/derived-table checks; existing contacts viewed separately.','checks':checks,'figures':records,'optional_helper':{'availability':'unavailable (script error)' if any(r['helper_returncode']==2 for r in records) else 'available','script_errors':sum(r['helper_returncode']==2 for r in records),'reason':'Project .venv lacks lxml; stderr preserved. No clean geometry claim is inferred from exit2. Independent PDF bounds plus visual inspection cover clipping/overlap.'},'helper_script':str(HELPER),'helper_script_sha256':sha(HELPER),'figure_manifest_sha256':sha(F/'qa/upgrade_figure_manifest.json'),'script_sha256':sha(__file__)}
(HERE/'H_final_figure_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v['passed']],'figures':len(records),'raw_heuristic_findings':sum(len(r['helper_findings']) for r in records)},ensure_ascii=False))
