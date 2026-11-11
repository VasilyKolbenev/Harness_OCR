"""Bounded raster decoder; XML never invents image coordinates."""
from pathlib import Path
import warnings
from .util import file_sha

def decode(path, output, settings):
    from PIL import Image, ImageOps
    path=Path(path); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if path.stat().st_size>settings.max_upload_bytes: raise ValueError('upload_size_limit')
    sha=file_sha(path)
    with path.open('rb') as f: magic=f.read(32)
    if magic.lstrip().startswith(b'<'): return _xml(path,sha)
    pages=[]
    def save(image,n):
        if n>settings.max_pages: raise ValueError('page_limit')
        image=ImageOps.exif_transpose(image)
        if image.width*image.height>settings.max_page_pixels: raise ValueError('page_pixel_limit')
        image=image.convert('RGB'); target=output/f'page-{n}.png'; image.save(target)
        pages.append({'page':n,'page_id':f'p{n}','path':str(target.resolve()),
                      'source_size':list(image.size),'raster_sha256':file_sha(target),
                      'transform':{'version':'raster-v1','orientation':'EXIF_transposed',
                                   'source_sha256':sha,'raster_sha256':file_sha(target)}})
    if magic.startswith(b'%PDF-'):
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(path) as pdf:
            if len(pdf)>settings.max_pages: raise ValueError('page_limit')
            for n in range(len(pdf)):
                page=pdf[n]
                try:
                    w,h=page.get_size(); scale=200/72
                    if w*h*scale*scale>settings.max_page_pixels: raise ValueError('pdf_raster_pixel_limit')
                    bitmap=page.render(scale=scale)
                    try: save(bitmap.to_pil(),n+1)
                    finally: bitmap.close()
                finally: page.close()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(path) as im:
                if getattr(im,'n_frames',1)>settings.max_pages: raise ValueError('page_limit')
                for n in range(getattr(im,'n_frames',1)):
                    im.seek(n); save(im,n+1)
    return {'source_sha256':sha,'pages':pages,'observations':[],
            'native_bridge':'raster_only_text_layer_not_trusted'}

def _xml(path,sha):
    from defusedxml import ElementTree as ET
    root=ET.parse(path).getroot(); observations=[]; count=0
    def walk(node,locator):
        nonlocal count
        count+=1
        if count>20000: raise ValueError('xml_node_limit')
        if node.text and node.text.strip():
            observations.append({'id':f'xml{count}','source_sha256':sha,'text':node.text,
               'evidence_kind':'structured_xml','xpath':locator,'namespace_map':{},
               'schema_version':'unqualified-generic-xml-v1',
               'structural_validation_status':'not_checked','completed':True,
               'capture_id':'xml-parser','model_id':'none'})
        for i,child in enumerate(node): walk(child,locator+'/'+str(child.tag)+f'[{i+1}]')
    walk(root,'/'+str(root.tag))
    return {'source_sha256':sha,'pages':[],'observations':observations,
            'native_bridge':'generic_xml_unqualified','signature_verified':False}
