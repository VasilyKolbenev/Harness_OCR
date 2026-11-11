"""Docker readiness probe: no tokens/document text on stdout or stderr."""
import json
import os
import urllib.request

def check_ready(base_url='http://127.0.0.1:8080',token=None):
    token=token or os.getenv('HARNESS_API_TOKEN','')
    if not token:return False
    request=urllib.request.Request(base_url.rstrip('/')+'/health/ready',
                     headers={'Authorization':'Bearer '+token})
    try:
        with urllib.request.urlopen(request,timeout=4) as response:
            body=response.read(1024*1024+1)
            if response.status!=200 or len(body)>1024*1024:return False
            value=json.loads(body)
        return (isinstance(value,dict) and value.get('ready') is True
                and value.get('fixture') is False)
    except Exception:return False

if __name__=='__main__':raise SystemExit(0 if check_ready() else 1)
