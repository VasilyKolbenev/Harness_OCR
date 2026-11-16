"""GPU delivery orchestration contracts. Fakes and non-neural processes only."""
from __future__ import annotations
import copy
from contextlib import redirect_stdout
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.gpu_delivery import plan,run,watchdog
from harness_ocr.profiles import ProfileRegistry
from harness_ocr.settings import Settings


def make_input(root,documents=1):
    docs=[]
    for index in range(documents):
        path=root/f'private-passport-{index}.xml';path.write_text('<fictional/>',encoding='utf-8')
        docs.append({'id':'Citizen_Ivanov_'+str(index),'path':path.name,'profile_id':'D07',
                     'subset':['payer.name']})
    manifest=root/'own.json';manifest.write_text(json.dumps({'schema_version':'harness-own-documents/1',
        'documents':docs}),encoding='utf-8')
    return manifest


class MetadataTests(unittest.TestCase):
    def test_four_default_arms_and_explicit_x_are_rotated_without_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=make_input(Path(folder),2);sha,docs=plan.load_documents(manifest)
            selected=plan.arms();rows=plan.schedule(docs,selected,sha)
            self.assertEqual(len(selected),4);self.assertEqual(len(rows),8)
            self.assertEqual([r['arm'] for r in rows[:4]],[a['id'] for a in selected])
            self.assertEqual([r['arm'] for r in rows[4:]],[a['id'] for a in selected[1:]+selected[:1]])
            self.assertEqual(len({r['row_id'] for r in rows}),8)
            self.assertEqual(len(plan.arms(include_x=True)),6)
            self.assertEqual(len(plan.arms('qwen-4b')),2)

    def test_metadata_preflight_accepts_missing_weights_native_and_avoids_cuda_or_children(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);output=root/'plan';stream=io.StringIO()
            args=['runner','--manifest',str(manifest),'--output',str(output),'--models',str(root/'missing')]
            imported_before=set(sys.modules)
            with patch('sys.argv',args),redirect_stdout(stream), \
                 patch.object(run,'model_dependencies',side_effect=AssertionError('no CUDA/model imports')), \
                 patch.object(run.subprocess,'Popen',side_effect=AssertionError('no child')), \
                 patch.object(run,'pin_models',side_effect=AssertionError('no weights')):
                self.assertEqual(run.main(),0)
            self.assertFalse((set(sys.modules)-imported_before)&{'torch','transformers','onnxruntime','ultralytics'})
            self.assertEqual(json.loads(stream.getvalue())['preflight_neural_calls'],0)
            value=plan.read(output/'plan.json')
            self.assertEqual(len(value['documents']),1);self.assertFalse(value['model_verification_performed'])
            self.assertIn('qwen-4b',value['required_model_keys'])
            self.assertEqual(value['documents'][0]['profile_id'],'payment_notice_v1')

    def test_invalid_profiles_subsets_paths_and_inline_gold_fail_before_model_work(self):
        for invalid in ('profile','subset','gold','outside'):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);manifest=make_input(root);value=plan.read(manifest)
                if invalid=='profile':value['documents'][0]['profile_id']='D999'
                elif invalid=='subset':value['documents'][0]['subset']=['invented.field']
                elif invalid=='gold':value['documents'][0]['gold']={'payer.name':'SECRET'}
                else:value['documents'][0]['path']='../outside.xml'
                manifest.write_text(json.dumps(value),encoding='utf-8')
                with self.assertRaises(ValueError):plan.load_documents(manifest)

    def test_strict_budgets_and_type_sensitive_exact_raw_do_not_normalize(self):
        for value in (False,True,0,86401,float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):plan.require_budget(value)
        self.assertFalse(plan.json_exact({'n':True},{'n':1}))
        self.assertFalse(plan.json_exact('Лeсников','Лесников'))
        result={'fields':{'x':{'extraction_status':'observed','raw_value':'12 34','validation_status':'pass'}}}
        self.assertEqual(plan.exact_score(result,{'x':'1234'})['wrong'],1)
        missing=plan.exact_score(None,{'x':'1234'})
        self.assertFalse(missing['measured']);self.assertNotIn('wrong',missing)

    def test_model_hash_and_deadline_guards_are_checked_without_loading_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'tiny').mkdir();path=root/'tiny/weights.bin';path.write_bytes(b'fixed')
            registry={'models':{'tiny':{'files':[{'file':'weights.bin','sha256':plan.file_sha(path)}]}}}
            pins,_=plan.pin_models(root,['tiny'],plan.utc()+timedelta(seconds=5),registry)
            self.assertEqual(pins['tiny/weights.bin']['sha256'],hashlib.sha256(b'fixed').hexdigest())
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'sha_mismatch'):plan.pin_models(root,['tiny'],None,registry)
            with self.assertRaises(plan.BudgetExpired):plan.file_sha(path,plan.utc()-timedelta(seconds=1))

    def test_separate_gold_schema_is_validated_before_nn_and_never_enters_plan_values(self):
        for bad in (None,'typed','unknown_doc','unknown_field','empty'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);manifest=make_input(root);gold=root/'gold.json'
                value={'schema_version':'harness-gpu-delivery-gold/1',
                       'documents':{'Citizen_Ivanov_0':{'payer.name':'REFERENCE_SECRET'}}}
                if bad=='typed':value['documents']['Citizen_Ivanov_0']['payer.name']={'items':[]}
                elif bad=='unknown_doc':value['documents']['outside']={'payer.name':'value'}
                elif bad=='unknown_field':value['documents']['Citizen_Ivanov_0']['payee.inn']='value'
                elif bad=='empty':value['documents']['Citizen_Ivanov_0']['payer.name']=''
                gold.write_text(json.dumps(value),encoding='utf-8')
                args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(root/'plan'),
                                              '--gold',str(gold)])
                with patch.object(run,'pin_models',side_effect=AssertionError('not before validation')):
                    if bad:
                        with self.assertRaises(ValueError):run.build_plan(args)
                    else:
                        planned=run.build_plan(args)
                        self.assertNotIn('REFERENCE_SECRET',json.dumps(planned))
                        self.assertTrue(planned['gold_metadata']['schema_validated_parent_only_before_inference'])
                        self.assertFalse(planned['gold_metadata']['reference_values_sent_to_worker'])

    def test_v1_string_gold_cannot_pretend_to_score_a_structured_typed_field(self):
        from harness_ocr.typed_fields import STRUCTURED_TYPES
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);value=plan.read(manifest)
            contract=ProfileRegistry().compile('D05')
            field=next(k for k,v in contract['field_definitions'].items() if v['value_type'] in STRUCTURED_TYPES)
            value['documents'][0].update(profile_id='D05',subset=[field])
            manifest.write_text(json.dumps(value),encoding='utf-8')
            gold=root/'gold.json';gold.write_text(json.dumps({'schema_version':'harness-gpu-delivery-gold/1',
                'documents':{'Citizen_Ivanov_0':{field:'raw-looking-but-typed'}}}),encoding='utf-8')
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(root/'plan'),'--gold',str(gold)])
            with self.assertRaisesRegex(ValueError,'cannot_score_structured_fields'):run.build_plan(args)

    def test_gold_with_only_missing_results_is_unmeasured_not_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);gold=root/'gold.json';out=root/'out';out.mkdir()
            gold.write_text(json.dumps({'schema_version':'harness-gpu-delivery-gold/1',
                'documents':{'Citizen_Ivanov_0':{'payer.name':'reference'}}}),encoding='utf-8')
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(out),'--gold',str(gold)])
            protocol=run.build_plan(args)
            result=run.score_after_inference(protocol,out,plan.utc()+timedelta(seconds=5))
            self.assertFalse(result['measured']);self.assertEqual(result['planned_reference_cells'],4)
            self.assertEqual(result['measured_reference_cells'],0)
            self.assertEqual(result['missing_result_reference_cells'],4)
            self.assertTrue(all('wrong' not in row['score'] for row in result['rows']))


class FakeOwner:
    live=0;created=0;closed=0;events=[];probe_device='cuda'
    def __init__(self,settings):
        type(self).live+=1;type(self).created+=1
        if type(self).live!=1:raise AssertionError('parallel owners')
        self.settings=settings;self.process=None;self.calls=[]
        self.events.append(('owner',str(settings.data_dir),settings.qwen,settings.rescue_text_detector))
    def call(self,op,args=None,**kwargs):
        self.calls.append(op)
        if kwargs.get('cancel') and kwargs['cancel']():raise plan.BudgetExpired('fake cancellation')
        return {'device':self.probe_device,'gpu':'fake-test-only','loaded_models':[
            'xiaomi',self.settings.qwen,'detector','cyrillic']}
    def close(self):type(self).closed+=1;type(self).live-=1


class FakeSampler:
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def result(self):return {'wall_seconds':1.0,'gpu_samples':1,
        'peak_sampled_attributable_gpu_bytes':200,'runner_plus_worker_peak_sampled_rss_bytes':100,
        'measurement_errors':[]}


class FakeEngine:
    runs=[]
    def __init__(self,settings,backend):self.settings=settings;self.backend=backend
    def run(self,jid,path,sha,profile_id,**kwargs):
        if 'shared_optics' in kwargs:raise AssertionError('no final optics sharing')
        self.runs.append((str(self.settings.data_dir),self.settings.qwen,self.settings.rescue_text_detector))
        self.settings.data_dir.mkdir(parents=True);plan.write(self.settings.data_dir/'state.json',{'job':jid})
        self.backend.call('select',{},cancel=kwargs['cancel'])
        contract=ProfileRegistry().compile(profile_id,subset=kwargs['subset'])
        return {'document_id':jid,'source_sha256':sha,'profile_id':profile_id,
            'profile_version':contract['profile_version'],'terminal_status':'completed','completed':True,
            'decision':'review','all_required_fields_validated':False,'native_ingestion':None,
            'fields':{key:{'raw_value':'SECRET_LITERAL','extraction_status':'observed','validation_status':'pass',
                           'decision':'review'} for key in kwargs['subset']},
            'calls':[{'op':'select','peak_cuda_allocated_bytes':123,'peak_cuda_reserved_bytes':456}],
            'optics':{'private':'SECRET_LITERAL'}}


class FakeProcess:
    pid=123;returncode=0
    def poll(self):return 0
    def wait(self,timeout=None):return 0


class RowTests(unittest.TestCase):
    def setUp(self):
        FakeOwner.live=FakeOwner.created=FakeOwner.closed=0;FakeOwner.events=[];FakeOwner.probe_device='cuda'
        FakeEngine.runs=[]
    def spec(self,root):
        manifest=make_input(root);sha,docs=plan.load_documents(manifest)
        arm=plan.arms('qwen-0.8b')[0];row=root/'row';row.mkdir()
        return {'row_id':'r1','row_dir':str(row),'document':docs[0],
            'settings':plan.settings_dict(plan.COMMON,arm,root/'models',root/'locks','placeholder'),
            'code_pins':{},'runner_sha256':'fixed','compute_deadline_utc':(plan.utc()+timedelta(seconds=60)).isoformat(),
            'worker_token_sha256':'unused'}

    def test_gold_whitelist_is_rejected_before_dependencies_and_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=self.spec(Path(folder));spec['gold']={'payer.name':'SECRET_LITERAL'}
            with patch.object(run,'model_dependencies',side_effect=AssertionError('cannot load')):
                with self.assertRaisesRegex(ValueError,'closed_inference_whitelist'):run.perform_row(spec)
            self.assertEqual(FakeOwner.created,0)

    def test_warmup_then_review_result_is_completed_and_closed_with_separate_costs(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=self.spec(Path(folder))
            with patch.object(run,'code_pins',return_value={}):
                code=run.perform_row(spec,(FakeEngine,FakeOwner,Settings,FakeSampler))
            self.assertEqual(code,0);self.assertEqual(FakeOwner.live,0);self.assertEqual(FakeOwner.closed,1)
            row=Path(spec['row_dir']);measurement=plan.read(row/'measurement.json')
            self.assertEqual(measurement['status'],'completed');self.assertTrue(measurement['supervisor_closed'])
            self.assertIn('document_resources',measurement);self.assertTrue((row/'warmup.json').exists())
            self.assertEqual(plan.read(row/'result.json')['decision'],'review')

    def test_cancel_before_owner_or_during_document_closes_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=self.spec(Path(folder));spec['compute_deadline_utc']=(plan.utc()-timedelta(seconds=1)).isoformat()
            run.perform_row(spec,(FakeEngine,FakeOwner,Settings,FakeSampler))
            self.assertEqual(FakeOwner.created,0)
            self.assertTrue(plan.read(Path(spec['row_dir'])/'measurement.json')['administrative_censor'])
        with tempfile.TemporaryDirectory() as folder:
            spec=self.spec(Path(folder))
            class CancelEngine(FakeEngine):
                def run(self,*args,**kwargs):
                    plan.write(self.settings.data_dir.parent/'parent-stop.json',{'reason':'test'})
                    if kwargs['cancel']():raise plan.BudgetExpired('test')
            with patch.object(run,'code_pins',return_value={}):run.perform_row(spec,(CancelEngine,FakeOwner,Settings,FakeSampler))
            self.assertEqual(FakeOwner.live,0);self.assertEqual(FakeOwner.created,FakeOwner.closed)

    def test_facade_does_not_restart_or_call_underlying_owner_after_failure(self):
        owner=SimpleNamespace(process=None,call=Mock(side_effect=RuntimeError('first failure')),close=Mock())
        facade=run.FailStopOwner(owner)
        with self.assertRaises(RuntimeError):facade.call('select',{})
        with self.assertRaisesRegex(RuntimeError,'no_retry'):facade.call('ocr',{})
        self.assertEqual(owner.call.call_count,1)
        self.assertEqual(facade.attempts,{'select':1});self.assertEqual(facade.refused_after_failure,1)
        facade.close();owner.close.assert_called_once()


class MatrixTests(unittest.TestCase):
    setUp=RowTests.setUp
    def test_fake_matrix_is_fresh_sequential_gold_free_and_public_report_has_no_literals(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);out=root/'comparison';gold=root/'gold.json'
            gold.write_text(json.dumps({'schema_version':'harness-gpu-delivery-gold/1',
                'documents':{'Citizen_Ivanov_0':{'payer.name':'OTHER_GOLD_SECRET'}}}),encoding='utf-8')
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(out),'--models',str(root/'models'),
                '--lock-dir',str(root/'locks'),'--gold',str(gold),'--total-seconds','120','--execute'])
            specs=[]
            def fake_popen(command,**kwargs):
                spec=plan.read(command[-1]);specs.append(copy.deepcopy(spec))
                run.perform_row(spec,(FakeEngine,FakeOwner,Settings,FakeSampler));return FakeProcess()
            tree=SimpleNamespace(records={},errors=[],alive=lambda:[])
            original_gold=run.load_gold_after_inference
            def score_only_closed(*args,**kwargs):
                self.assertEqual(FakeOwner.live,0);return original_gold(*args,**kwargs)
            with patch.object(run,'pin_models',return_value=({},'registrysha')), \
                 patch.object(run,'code_pins',return_value={}),patch.object(run,'validate_runtime_pins'), \
                 patch.object(run.subprocess,'Popen',side_effect=fake_popen), \
                 patch.object(run,'VerifiedTree',return_value=tree), \
                 patch.object(run,'load_gold_after_inference',side_effect=score_only_closed):
                result=run.execute(args)
            self.assertTrue(result['technical_execution_passed']);self.assertEqual(result['planned_rows'],4)
            self.assertEqual(FakeOwner.created,4);self.assertEqual(FakeOwner.closed,4)
            self.assertEqual(len({s['row_dir'] for s in specs}),4);self.assertEqual(len(FakeEngine.runs),4)
            for spec in specs:
                self.assertEqual(set(spec),run.SPEC_KEYS)
                self.assertNotIn('OTHER_GOLD_SECRET',json.dumps(spec));self.assertNotIn('gold_metadata',spec)
            public=json.dumps(result,ensure_ascii=False)
            for private in ('SECRET_LITERAL','OTHER_GOLD_SECRET','Citizen_Ivanov','private-passport',str(root)):
                self.assertNotIn(private,public)
            self.assertTrue(all(r['review_fields']==1 for r in result['rows']))
            self.assertTrue(all(r['score']['wrong']==1 for r in result['quality']['rows']))
            self.assertTrue(all(r['max_call_cuda_reserved_bytes']==456 for r in result['rows']))
            self.assertFalse(result['quality_qualified'])

    def test_failed_cuda_probe_stops_without_cpu_fallback_and_missing_pairs_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);out=root/'comparison'
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(out),
                '--lock-dir',str(root/'locks'),'--total-seconds','120','--execute'])
            FakeOwner.probe_device='cpu'
            def fake_popen(command,**kwargs):
                run.perform_row(plan.read(command[-1]),(FakeEngine,FakeOwner,Settings,FakeSampler));return FakeProcess()
            tree=SimpleNamespace(records={},errors=[],alive=lambda:[])
            with patch.object(run,'pin_models',return_value=({},'registrysha')), \
                 patch.object(run,'code_pins',return_value={}),patch.object(run,'validate_runtime_pins'), \
                 patch.object(run.subprocess,'Popen',side_effect=fake_popen),patch.object(run,'VerifiedTree',return_value=tree):
                result=run.execute(args)
            self.assertFalse(result['technical_execution_passed']);self.assertEqual(result['recorded_rows'],1)
            self.assertEqual(result['missing_rows'],3);self.assertEqual(FakeOwner.created,1)
            self.assertEqual(FakeOwner.closed,1);self.assertEqual(FakeEngine.runs,[])

    def test_no_gold_review_and_null_nvml_are_not_fake_accuracy_or_zero_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root);out=root/'comparison'
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(out),'--qwen','qwen-0.8b',
                '--lock-dir',str(root/'locks'),'--total-seconds','120','--execute'])
            class NoNvml(FakeSampler):
                def result(self):return {**super().result(),'gpu_samples':0,
                    'peak_sampled_attributable_gpu_bytes':None,'measurement_errors':['NVML_unavailable:ImportError']}
            def fake_popen(command,**kwargs):
                run.perform_row(plan.read(command[-1]),(FakeEngine,FakeOwner,Settings,NoNvml));return FakeProcess()
            tree=SimpleNamespace(records={},errors=[],alive=lambda:[])
            with patch.object(run,'pin_models',return_value=({},'registrysha')), \
                 patch.object(run,'code_pins',return_value={}),patch.object(run,'validate_runtime_pins'), \
                 patch.object(run.subprocess,'Popen',side_effect=fake_popen),patch.object(run,'VerifiedTree',return_value=tree):
                result=run.execute(args)
            self.assertTrue(result['technical_execution_passed']);self.assertTrue(result['correctness_unmeasured'])
            self.assertFalse(result['quality']['measured']);self.assertFalse(result['resource_measurement_complete'])
            self.assertIsNone(result['rows'][0]['document_resources']['peak_sampled_attributable_gpu_bytes'])
            self.assertEqual(result['rows'][0]['max_call_cuda_allocated_bytes'],123)
            self.assertEqual(result['rows'][0]['max_call_cuda_reserved_bytes'],456)

    def test_native_no_generation_is_not_reported_as_gpu_document_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result={'completed':True,'decision':'review','fields':{},
                'native_ingestion':{'adapter_id':'EMCHD_1'},'calls':[{'op':'ingest'}]}
            plan.write(root/'result.json',result)
            public=run.safe_row({'row_id':'r','public_document_id':'doc_opaque','arm':'db'},root,{})
            self.assertTrue(public['native_ingestion_present']);self.assertTrue(public['zero_generation_document'])
            self.assertEqual(public['model_generation_calls'],0)

    def test_missing_weights_keep_every_known_pair_explicit_without_creating_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_input(root)
            args=run.parser().parse_args(['--manifest',str(manifest),'--output',str(root/'out'),
                '--lock-dir',str(root/'locks'),'--total-seconds','120','--execute'])
            with patch.object(run,'pin_models',side_effect=ValueError('required_model_file_missing')), \
                 patch.object(run.subprocess,'Popen',side_effect=AssertionError('no owner')):
                result=run.execute(args)
            self.assertEqual(result['planned_rows'],4);self.assertEqual(result['missing_rows'],4)
            self.assertEqual(result['recorded_rows'],0);self.assertFalse(result['technical_execution_passed'])
            self.assertEqual(FakeOwner.created,0)


class WatchdogTests(unittest.TestCase):
    def test_real_short_lived_owner_disappearance_is_not_integrity_failure(self):
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'])
        tree=watchdog.VerifiedTree(process.pid);original=tree.verified
        def end_between_checks(record):
            value=original(record)
            if value is not None:process.wait(timeout=5)
            return value
        with patch.object(tree,'verified',side_effect=end_between_checks):tree.snapshot()
        self.assertEqual(tree.errors,[]);self.assertEqual(tree.alive(),[])

    def test_reused_pid_is_not_killed_but_is_strict_integrity_error(self):
        process=SimpleNamespace(create_time=lambda:2,is_running=lambda:True,status=lambda:'running',
            terminate=Mock(),kill=Mock())
        tree=watchdog.VerifiedTree.__new__(watchdog.VerifiedTree)
        tree.psutil=SimpleNamespace(Process=lambda pid:process,NoSuchProcess=ProcessLookupError,Error=OSError,STATUS_ZOMBIE='zombie')
        tree.errors=[]
        self.assertIsNone(tree.verified({'pid':10,'created':1}))
        self.assertIn('pid_identity_mismatch',tree.errors)
        process.kill.assert_not_called();process.terminate.assert_not_called()

    def test_hard_cleanup_kills_owned_descendants_and_keeps_unrelated_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ready=root/'child.json';script=root/'owner.py'
            script.write_text('import json,subprocess,sys,time\nfrom pathlib import Path\n'
                'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\n'
                'Path('+repr(str(ready))+').write_text(json.dumps({"pid":p.pid}))\n'
                'time.sleep(30)\n',encoding='utf-8')
            bystander=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            process=subprocess.Popen([sys.executable,str(script)],start_new_session=os.name=='posix')
            tree=watchdog.VerifiedTree(process.pid,new_session=os.name=='posix')
            try:
                import time
                until=time.monotonic()+5
                while not ready.exists() and process.poll() is None and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(ready.exists());tree.snapshot();self.assertGreaterEqual(len(tree.records),2)
                actions=tree.stop(wait_seconds=.1);process.wait(timeout=5)
                self.assertTrue(actions);self.assertEqual(tree.alive(),[])
                self.assertIsNone(bystander.poll())
                self.assertNotIn(bystander.pid,tree.records)
            finally:
                tree.stop(wait_seconds=.05)
                if process.poll() is None:process.kill()
                process.wait(timeout=5);bystander.terminate();bystander.wait(timeout=5)


if __name__=='__main__':unittest.main()
