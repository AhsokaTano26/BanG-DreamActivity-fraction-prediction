"""Verify the compiled paper, fonts, page bounds and supplied scientific figures."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

P=Path(__file__).resolve().parents[1]
Q=P/'qa'


def tool(name):
    return shutil.which(name) or '/opt/homebrew/bin/'+name


def run(name,*args):
    return subprocess.check_output([tool(name),*map(str,args)],cwd=P,text=True)


def main():
    checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
        assert value,name
    pdf=P/'output/main.pdf';alternate=P/'output/中文论文.pdf'
    check('Both PDF delivery names have identical bytes',pdf.read_bytes()==alternate.read_bytes())
    info=run('pdfinfo',pdf);(Q/'pdfinfo.txt').write_text(info)
    check('Author metadata is Tano',bool(re.search(r'^Author:\s+Tano$',info,re.M)))
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
    check('Fifteen page manuscript',pages==15)
    log=(P/'output/main.log').read_text()
    check('Final XeLaTeX pass has no warnings or box overflow',not re.search(r'Warning|Overfull|Underfull|Missing character|! ',log))
    check('All bibliography entries rendered',(P/'output/main.bbl').read_text().count(r'\bibitem')==11)
    plain=run('pdftotext','-layout',pdf,'-')
    (Q/'paper_extracted.txt').write_text(plain)
    check('Readable text without replacement glyphs','\ufffd' not in plain and 'Tano' in plain and '103' in plain and '0.915' in plain and '0.471' in plain)
    fonts=run('pdffonts',pdf);(Q/'embedded_fonts.txt').write_text(fonts)
    rows=[line.split() for line in fonts.splitlines()[2:] if line.strip()]
    check('Every manuscript and figure font embedded',all(re.search(r'\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',line) for line in fonts.splitlines()[2:] if line.strip()))
    check('No Type 3 fonts','Type 3' not in fonts)
    bbox=run('pdftotext','-bbox',pdf,'-')
    root=ET.fromstring(bbox);xmlpages=root.findall('.//{*}page')
    check('All pages have text bounds',len(xmlpages)==pages)
    bounds=[]
    for i,page in enumerate(xmlpages,1):
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        words=page.findall('.//{*}word')
        check('A4 geometry page '+str(i),abs(width-595.276)<1 and abs(height-841.89)<1)
        check('Content present page '+str(i),len(words)>12)
        valid=all(float(w.attrib['xMin'])>=55 and float(w.attrib['xMax'])<=width-55 and float(w.attrib['yMin'])>=45 and float(w.attrib['yMax'])<=height-30 for w in words)
        check('Text inside page and print margins page '+str(i),valid)
        bounds.append({'page':i,'words':len(words),'min_x':min(float(w.attrib['xMin']) for w in words),'max_x':max(float(w.attrib['xMax']) for w in words)})
    records=json.loads((Q/'figure_records.json').read_text())
    check('Seven figure groups',len(records)==7)
    for record in records:
        name=record['name'];v=P/'figures'/f'{name}.pdf';p=P/'figures'/f'{name}.png'
        f=run('pdffonts',v)
        check(name+' vector fonts embedded',all('yes' in line for line in f.splitlines()[2:] if line.strip()) and 'Type 3' not in f)
        with Image.open(p) as im:
            check(name+' 400 dpi preview',min(im.info.get('dpi',(0,0)))>=399 and im.width>=1500 and im.height>=800)
        check(name+' grayscale preview exists',(Q/f'{name}_gray.png').exists())
    highlights=(P/'highlights.txt').read_text().splitlines()
    check('Five concise highlights',len(highlights)==5 and all(len(s)<=85 for s in highlights))
    references=json.loads((P/'data/reference_sources.json').read_text())
    check('Eleven inspectable primary source links',len(references)==11 and all(s['url'].startswith('https://') for s in references))
    result={'status':'passed','checks':len(checks),'pages':pages,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'page_bounds':bounds,'results':checks,'visual_review':'See qa/visual_review.json for human-model inspection of the rendered pages and grayscale scientific figures.'}
    (Q/'artifact_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks','pages','pdf_sha256']}))


if __name__=='__main__':
    main()
