import argparse
from pathlib import Path
import json
from .settings import Settings
from .util import file_sha,write_json

def main():
    p=argparse.ArgumentParser(prog='harness-ocr');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('profiles');sub.add_parser('mcp')
    prep=sub.add_parser('prepare-models');prep.add_argument('--models',type=Path,required=True)
    prep.add_argument('--only',nargs='+',default=['xiaomi','detector','cyrillic','qwen-0.8b','qwen-4b'])
    prep.add_argument('--local-sources',type=Path);prep.add_argument('--download',action='store_true')
    verify=sub.add_parser('verify-models');verify.add_argument('--models',type=Path,required=True)
    verify.add_argument('--only',nargs='+',default=['xiaomi','detector','cyrillic','qwen-0.8b','qwen-4b'])
    infer=sub.add_parser('infer');infer.add_argument('--input',type=Path,required=True)
    infer.add_argument('--profile',required=True);infer.add_argument('--output',type=Path,required=True)
    infer.add_argument('--subset',nargs='+');infer.add_argument('--qwen',choices=['qwen-0.8b','qwen-4b'],default='qwen-0.8b')
    infer.add_argument('--mode',choices=['hybrid','rules'],default='hybrid')
    bench=sub.add_parser('bench');bench.add_argument('--manifest',type=Path,required=True);bench.add_argument('--output',type=Path,required=True)
    bench.add_argument('--no-controller',action='store_true')
    for command in [infer,bench]:
        command.add_argument('--models',type=Path,required=True);command.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        command.add_argument('--max-new-tokens',type=int,default=1024);command.add_argument('--call-timeout',type=float,default=120)
        command.add_argument('--document-timeout',type=float,default=600)
    replay=sub.add_parser('replay-candidates');replay.add_argument('--record',type=Path,required=True)
    replay.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='profiles':
        from .profiles import ProfileRegistry
        value=ProfileRegistry().list()
    elif args.command in {'prepare-models','verify-models'}:
        from .models import prepare_models,verify_models
        if args.command=='prepare-models':
            sources=json.loads(args.local_sources.read_text(encoding='utf-8-sig')) if args.local_sources else {}
            value=prepare_models(args.models,args.only,sources,args.download)
        else:value=verify_models(args.models,args.only)
    elif args.command=='infer':
        from .engine import Engine
        from .models import verify_models
        from .inference.supervisor import Supervisor
        if args.output.exists():raise ValueError('output_already_exists')
        settings=_settings(args,args.output.parent/'state')
        checked=verify_models(args.models,['xiaomi','detector','cyrillic',args.qwen])
        if not checked['ok']:raise ValueError(str(checked['errors']))
        backend=Supervisor(settings)
        try:
            backend.call('probe',{'warmup':True});engine=Engine(settings,backend)
            import uuid
            value=engine.run(uuid.uuid4().hex,args.input,file_sha(args.input),args.profile,
                subset=args.subset,qwen=args.qwen,mode=args.mode)
            value.pop('optics',None);write_json(args.output,value)
        finally:backend.close()
    elif args.command=='bench':
        from .benchmark import run_benchmark
        value=run_benchmark(args.manifest,args.output,_settings(args,args.output/'state'),include_controller=not args.no_controller)
    elif args.command=='replay-candidates':
        from .core.trace import replay
        from .grounding import build_candidates
        record=json.loads(args.record.read_text(encoding='utf-8-sig'))
        if record['stage']!='candidates':raise ValueError('only_candidate_stage_supported')
        value=replay(record,lambda inputs,config:build_candidates(inputs['observations'],inputs['contract']))
        write_json(args.output,value)
    elif args.command=='mcp':
        from .mcp_server import main as run
        run();return
    print(json.dumps(value,ensure_ascii=False,indent=2))

def _settings(args,data_dir):
    return Settings(Path(data_dir).resolve(),args.models.resolve(),device=args.device,
      dtype='float32' if args.device=='cpu' else 'bfloat16',max_new_tokens=args.max_new_tokens,
      call_timeout_seconds=args.call_timeout,document_timeout_seconds=args.document_timeout,
      qwen=getattr(args,'qwen','qwen-0.8b'))

if __name__=='__main__':main()
