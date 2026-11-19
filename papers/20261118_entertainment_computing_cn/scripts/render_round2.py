"""Render every current PDF page and labeled contact sheets for visual inspection."""
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
import argparse,subprocess,json,hashlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper-root',required=True,type=Path);a=p.parse_args();P=a.paper_root.resolve();R=P/'qa/round2_render';R.mkdir(parents=True,exist_ok=True);report={}
for kind in ['main','supplement']:
 out=R/kind;out.mkdir(exist_ok=True)
 for stale in out.glob('page-*.png'):stale.unlink()
 subprocess.run(['pdftoppm','-r','96','-png',str(P/'output'/f'{kind}.pdf'),str(out/'page')],check=True)
 pages=sorted(out.glob('page-*.png'))
 # 6 pages per contact sheet, including explicit manuscript page labels.
 sheets=[]
 for start in range(0,len(pages),6):
  subset=pages[start:start+6];canvas=Image.new('RGB',(1230,1180),'#dddddd');draw=ImageDraw.Draw(canvas)
  for i,path in enumerate(subset):
   im=Image.open(path).convert('RGB');im.thumbnail((390,540))
   x=15+(i%3)*410;y=30+(i//3)*590
   canvas.paste(im,(x,y));draw.text((x,y-20),f'{kind} page {start+i+1}',fill='black')
  name=R/f'{kind}_contact_{start//6+1:02d}.png';canvas.save(name);sheets.append(str(name.relative_to(P)))
 report[kind]={'pages':len(pages),'contact_sheets':sheets,'page_sha256':{str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in pages}}
(P/'qa/round2_render_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({k:v['pages'] for k,v in report.items()})
