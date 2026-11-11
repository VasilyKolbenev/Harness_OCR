"""Create explicitly synthetic DEV transport/inference fixtures, not qualification data."""
from pathlib import Path
import argparse
import json
from PIL import Image,ImageDraw,ImageFont,ImageEnhance

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path);args=p.parse_args()
    if args.output.exists():raise ValueError('output_must_be_new')
    candidates=[args.font,Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('C:/Windows/Fonts/arial.ttf')]
    font=next((x for x in candidates if x and x.is_file()),None)
    if font is None:raise RuntimeError('Cyrillic TrueType font required; use --font')
    args.output.mkdir(parents=True)
    im=Image.new('RGB',(1400,450),'white');draw=ImageDraw.Draw(im)
    draw.text((35,40),'СИНТЕТИЧЕСКИЙ DEV-ПРИМЕР. НЕ ДОКУМЕНТ.',font=ImageFont.truetype(str(font),26),fill='black')
    draw.text((35,150),'Номер паспорта: 123456',font=ImageFont.truetype(str(font),48),fill='black')
    draw.text((35,270),'Все сведения вымышлены. Только проверка конвейера.',font=ImageFont.truetype(str(font),26),fill='black')
    im.save(args.output/'clean.png')
    ImageEnhance.Contrast(im.rotate(1.2,fillcolor='white')).enhance(.75).save(args.output/'photo-like.jpg',quality=75)
    manifest={'schema_version':'harness-bench/1','split':'dev',
      'purpose':'Synthetic smoke only. One original, two correlated variants; no quality qualification.',
      'documents':[{'id':name,'original_id':'synthetic-number-one-original',
        'path':path,'profile_id':'D01','subset':['passport.number'],
        'gold':{'passport.number':'123456'}} for name,path in [('clean','clean.png'),('degraded','photo-like.jpg')]]}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output/'manifest.json')

if __name__=='__main__':main()
