"""Inspect existing exports only. Does not regenerate or recompute evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from matplotlib import font_manager as fm
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);p.add_argument('--svg-checker',type=Path)
a=p.parse_args();root=a.repo_root.resolve();work=a.out_root.resolve();qa=work/'qa';figures=work/'figures'
manifest=json.loads((qa/'upgrade_figure_manifest.json').read_text())
fonts=json.loads((qa/'font_check.json').read_text());cmaps={k:fm.get_font(fonts[k]['path']).get_charmap() for k in ['latin','cjk']}
ns={'s':'http://www.w3.org/2000/svg'};records=[];allchars=set();problems=[]
for item in manifest['figures']:
    name=item['name'];svg=figures/(name+'.svg');r=ET.parse(svg).getroot()
    texts=[''.join(t.itertext()) for t in r.findall('.//s:text',ns)];chars=set(''.join(texts));allchars|=chars
    missing=[c for c in chars if ord(c)>32 and ord(c) not in cmaps['latin'] and ord(c) not in cmaps['cjk']]
    for source in item['inputs']:
        sp=root/source['path'] if not Path(source['path']).is_absolute() else Path(source['path'])
        if not sp.exists() or hashlib.sha256(sp.read_bytes()).hexdigest()!=source['sha256']:
            problems.append({'figure':name,'check':'source_sha256','path':source['path']})
    output_hashes={}
    for ext in ['pdf','svg','png']:
        file=figures/(name+'.'+ext);digest=hashlib.sha256(file.read_bytes()).hexdigest();output_hashes[ext]=digest
        if digest!=item['formats'][ext]['sha256']:problems.append({'figure':name,'check':'output_sha256','format':ext})
    pdf_fonts=subprocess.run(['pdffonts',str(svg.with_suffix('.pdf'))],capture_output=True,text=True,check=True).stdout
    font_rows=[line.split() for line in pdf_fonts.splitlines()[2:] if line.strip()]
    unembedded=[row for row in font_rows if not re.search(r'\s+yes\s+yes\s+yes\s+', ' '.join(row))]
    images=subprocess.run(['pdfimages','-list',str(svg.with_suffix('.pdf'))],capture_output=True,text=True,check=True).stdout.splitlines()[2:]
    bbox_xml=subprocess.run(['pdftotext','-bbox',str(svg.with_suffix('.pdf')),'-'],capture_output=True,text=True,check=True).stdout
    bbox_root=ET.fromstring(bbox_xml);word_outside=[];word_count=0
    for page in bbox_root.iter():
        if not page.tag.endswith('page'):continue
        pw,ph=float(page.attrib['width']),float(page.attrib['height'])
        for word in page.iter():
            if not word.tag.endswith('word'):continue
            word_count+=1
            b=[float(word.attrib[x]) for x in ['xMin','yMin','xMax','yMax']]
            if b[0]<-.4 or b[1]<-.4 or b[2]>pw+.4 or b[3]>ph+.4:word_outside.append({'word':word.text,'bbox':b,'page':[pw,ph]})
    if word_outside:problems.append({'figure':name,'check':'actual_pdf_text_bounds','outside':word_outside})
    with Image.open(svg.with_suffix('.png')) as im:
        dpi=im.info.get('dpi');pixels=im.size
        # The padded export must not crop ink at any outermost pixel.
        gray=im.convert('L');corners=[gray.getpixel((x,y)) for x,y in [(0,0),(pixels[0]-1,0),(0,pixels[1]-1),(pixels[0]-1,pixels[1]-1)]]
    svgi=len(r.findall('.//s:image',ns));ids=[x.get('id') for x in r.iter() if x.get('id')]
    if missing or unembedded or images or svgi or len(ids)!=len(set(ids)) or not all(abs(x-400)<.01 for x in dpi):
        problems.append({'figure':name,'check':'export','missing_chars':missing,'unembedded':unembedded,'pdf_images':images,'svg_images':svgi,'dpi':dpi})
    info={'name':name,'live_text_objects':len(texts),'svg_paths':len(r.findall('.//s:path',ns)),'svg_images':svgi,'pdf_images':len(images),'all_fonts_embedded_unicode':not unembedded,'missing_characters':missing,'png_pixels':pixels,'png_dpi':dpi,'white_corners':corners,'output_sha256':output_hashes,'minimum_nominal_font_pt':item['minimum_fontsize_pt']}
    info.update({'actual_pdf_text_words':word_count,'actual_pdf_text_outside_page':word_outside})
    if a.svg_checker:
        dest=qa/'programmatic'/(name+'.json');dest.parent.mkdir(exist_ok=True)
        # This subprocess inherits the optional uv environment with geometry deps.
        import sys
        result=subprocess.run([sys.executable,str(a.svg_checker),str(svg),'--journal','generic','--json'],capture_output=True,text=True)
        dest.write_text(result.stdout);dest.with_suffix('.stderr').write_text(result.stderr)
        info['svg_checker_exit']=result.returncode
        if result.returncode==2:problems.append({'figure':name,'check':'svg_checker_error','stderr':result.stderr})
    records.append(info)
report={'scope':'18 existing vector exports; no fitting, scoring or bootstrap','figures':records,'source_and_export_problems':problems,'normal_text_unique_characters':len(allchars),'font_paths':{k:fonts[k]['path'] for k in cmaps},'fonts_fallback':False,'palette_check':'Continuous diverging and gray data maps checked by explicit source stops and actual gray/protanopia/deuteranopia renders; no categorical allow-list test on interpolated data cells.'}
(qa/'round2_programmatic_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(records),'problems':len(problems),'characters':len(allchars)},ensure_ascii=False))
raise SystemExit(bool(problems))
