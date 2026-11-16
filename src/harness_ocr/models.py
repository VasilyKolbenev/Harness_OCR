"""Pinned, verified model preparation. Never downloads on an inference request."""
from importlib import resources
from pathlib import Path
import json
import shutil
import urllib.request
import os
from urllib.parse import urlsplit
from .util import file_sha, digest, inside

def model_lock():
    return json.loads(resources.files('harness_ocr.data').joinpath('model_lock.json').read_text(encoding='utf-8'))

def model_download_url(spec, item):
    """Only the pinned official GitHub release or existing HF revision route."""
    if spec.get('source') == 'github-release':
        url = item.get('download_url', '')
        parts = urlsplit(url)
        expected = f'/ultralytics/assets/releases/download/{spec["revision"]}/{item["file"]}'
        if (parts.scheme != 'https' or parts.netloc != 'github.com' or
                parts.path != expected or parts.query or parts.fragment):
            raise ValueError('unapproved_model_release_url')
        return url
    return f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{item["file"]}'

def verify_models(root, names):
    lock=model_lock(); errors=[]; checked=[]
    for name in names:
        spec=lock['models'][name]
        for item in spec['files']:
            p=inside(Path(root)/name,item['file'])
            if not p.is_file(): errors.append(f'{name}/{item["file"]}:missing')
            elif file_sha(p)!=item['sha256']: errors.append(f'{name}/{item["file"]}:sha_mismatch')
            else: checked.append(f'{name}/{item["file"]}')
    return {'ok':not errors,'errors':errors,'checked_files':len(checked),
            'model_bundle_sha256':digest({k:lock['models'][k] for k in names})}

def prepare_models(root, names, local_sources=None, download=False):
    lock=model_lock(); local_sources=local_sources or {}; actions=[]
    for name in names:
        spec=lock['models'][name]; dest=Path(root)/name; dest.mkdir(parents=True,exist_ok=True)
        for item in spec['files']:
            target=inside(dest,item['file'])
            if target.is_file() and file_sha(target)==item['sha256']: continue
            temp=target.with_name(target.name+'.part')
            src=Path(local_sources[name])/item['file'] if name in local_sources else None
            if src and src.is_file() and file_sha(src)==item['sha256']:
                shutil.copyfile(src,temp)
            elif download:
                url=model_download_url(spec,item)
                with urllib.request.urlopen(url,timeout=120) as r, temp.open('wb') as f:
                    shutil.copyfileobj(r,f,8*1024*1024)
            else: raise FileNotFoundError(f'{name}/{item["file"]}: use --download or verified local source')
            if file_sha(temp)!=item['sha256']:
                raise ValueError(f'model_checksum_mismatch:{name}/{item["file"]}')
            os.replace(temp,target); actions.append(str(target))
    return {'actions':actions,**verify_models(root,names)}
