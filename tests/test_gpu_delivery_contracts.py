"""GPU delivery orchestration proofs with explicit fake NN outputs, never models."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

from harness_ocr.inference.locks import FileLease,LeaseBusy
from harness_ocr.inference.supervisor import ModelCanceled,ModelDeadline
from harness_ocr.settings import Settings
from harness_ocr.util import digest

ROOT=Path(__file__).resolve().parents[1]
def script(name):
    spec=importlib.util.spec_from_file_location('delivery_'+name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
smoke=script('gpu_smoke');locks=script('init_shared_locks')


class FakeSampler:
    def __init__(self,owner,device):self.owner=owner;self.device=device
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def result(self):return {'fixture_only':True,'peak_sampled_attributable_gpu_bytes':None}


class FakeOwner:
    """Image-backed fixtures labelled fake; they can never qualify a target GPU."""
    def __init__(self,settings,specs,*,failure=None,mutate=None,on_call=None):
        self.settings=settings;self.specs=specs;self.failure=failure;self.mutate=mutate;self.on_call=on_call
        self.calls=[];self.closed=False
    def generation(self,name,*,images=False):
        devices={'input_ids':'cuda:0','attention_mask':'cuda:0'}
        dtypes={'input_ids':'torch.int64','attention_mask':'torch.int64'}
        if images:devices.update(pixel_values='cuda:0');dtypes.update(pixel_values='torch.float32')
        return {'model_id':name,'device':'cuda','dtype':'bfloat16','completed':True,'stop_reason':'eos',
            'fresh_generation_state':True,'input_tokens':100,'generated_tokens':30,
            'input_ids_sha256':digest('fake-input-ids'),'effective_eos_token_ids':[7],
            'input_tensor_devices':devices,'input_tensor_dtypes':dtypes,
            'effective_parameter_devices':['cuda:0'],
            'effective_parameter_dtypes':['torch.bfloat16','torch.float32'],
            'json_constraint_applied':name!='xiaomi','seconds':.01}
    def call(self,op,args,**options):
        self.calls.append((op,deepcopy(args),options))
        report_path=self.settings.data_dir.parents[1]/'report.json'
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report['status']!='running' or not report['arms'][-1]['steps']:
            raise AssertionError('durable_attempt_missing_before_call')
        # The coordinator lease spans model-owner replacement, not each row.
        with unittest.TestCase().assertRaises(LeaseBusy):
            FileLease(self.settings.model_lock_dir/'gpu-delivery.lock').acquire()
        if self.on_call:self.on_call(op)
        if self.failure and self.failure[0]==op:raise self.failure[1]
        from harness_ocr.inference.constraints import build_response_schema,ENGINE
        from harness_ocr.inference.vision import prepare_visual_regions,blind_prompt,parse_visual_output,VERSION
        if op=='probe':
            return {'device':'cuda','gpu':'EXPLICIT_FAKE_GPU','capability':[12,0],
                    'loaded_models':['xiaomi',self.settings.qwen,'detector','cyrillic']}
        if op in {'cheap','cheap_rescue'}:
            page=args['page'];variant='db' if op=='cheap' else self.settings.rescue_text_detector
            observation={'id':variant+'_fixture','source_sha256':args['source_sha256'],
                'text':'Номер паспорта: 777777','evidence_kind':'image_region','completed':True,
                'page':1,'page_id':page['page_id'],'source_size':page['source_size'],
                'bbox':[0,0,*page['source_size']]}
            diag={'raster_sha256':page['raster_sha256'],'runtime':{'text_detector':variant,'recognizer_device':'cpu'},
                  'lines_processed':1,'ctc':[{'actual_output_shape':[1,40,852]}]}
            if variant!='db':diag['detector']={'device':'cuda','cuda_synchronized':True,
                'checkpoint_sha256':self.specs[variant+'-text']['files'][0]['sha256'],
                'coordinates_refer_to':'actual_page_raster_pixels'}
            region={'id':variant+'_region','page':1,'page_id':page['page_id'],
                    'source_size':page['source_size'],'bbox':[0,0,*page['source_size']]}
            result={'completed':True,'observations':[observation],'regions':[region],'diagnostics':diag}
        elif op=='ocr':
            result={**self.generation('xiaomi',images=True),'text':'Номер паспорта: 999999',
                    'input_image_sha256':args['expected_image_sha256'],'input_image_expected_sha_verified':True}
        elif op=='select':
            result={**self.generation(args['name']),'text':'{"proposals":[],"reread_requests":[]}',
                    'response_schema_sha256':digest(build_response_schema(args['prompt'],'selector')),
                    'constraint_engine':ENGINE}
        elif op=='visual_read':
            _,provenance=prepare_visual_regions(args['regions'])
            prompt=blind_prompt([p['region_id'] for p in provenance]);name=args['name']
            result={**self.generation(name,images=True),
                'text':'{"readings":{"r1":{"status":"readable","literal":"INCORRECT BUT WELL-FORMED FIXTURE TEXT"}}}',
                'purpose':'blind_optical_capture','modality':'vision','adapter_version':VERSION,
                'processor_class':'Qwen3VLProcessor','proposed_value_supplied':False,'ground_truth_supplied':False,
                'automatic_acceptance':False,'statistical_independence_claimed':False,'provenance':provenance,
                'blind_prompt':prompt,'blind_prompt_sha256':digest(prompt),
                'model_bundle_sha256':digest(self.specs[name]),'constraint_engine':ENGINE,
                'response_schema_sha256':digest(build_response_schema(prompt,'visual_read')),
                'pixel_values_sha256':digest('explicit-fake-image-tensor'),
                'image_grid_thw':[[1,8,40]],'pixel_values_shape':[320,1536]}
            result['captures']=parse_visual_output(result,provenance)
        else:raise AssertionError(op)
        if self.mutate:self.mutate(op,result)
        return result
    def close(self):self.closed=True


class GpuSmokeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.lock_dir=self.root/'locks';self.lock_dir.mkdir()
        for name in smoke.LOCK_FILES:(self.lock_dir/name).touch()
        names=['xiaomi','detector','cyrillic',*smoke.QWENS,'yolo26l-text','yolo26x-text']
        self.specs={name:{'files':[{'file':'explicit-fixture','sha256':digest(name)}]} for name in names}
        self.patcher=patch('harness_ocr.models.model_lock',return_value={'models':self.specs})
        self.patcher.start();self.addCleanup(self.patcher.stop);self.owners=[]
    def run_smoke(self,*,failure=None,mutate=None,on_call=None,folder='run',**options):
        def factory(settings):
            self.assertTrue(all(owner.closed for owner in self.owners),'Two owners overlapped')
            owner=FakeOwner(settings,self.specs,failure=failure,mutate=mutate,on_call=on_call)
            self.owners.append(owner);return owner
        return smoke.run_smoke(self.root/'models',self.root/folder,self.lock_dir,
            _owner_factory=factory,_verify_models=lambda *a:{'ok':True,'fixture_only':True},
            _sampler_factory=FakeSampler,**options)
    def test_fresh_sequential_owners_exercise_six_generations_without_accuracy_claim(self):
        report=self.run_smoke()
        self.assertEqual(report['status'],'pass',report.get('error'))
        self.assertEqual(report['neural_generation_calls_attempted'],6)
        self.assertEqual(report['completed_generation_calls'],6)
        self.assertEqual(len(self.owners),2);self.assertTrue(all(o.closed for o in self.owners))
        self.assertEqual([[op for op,_,_ in o.calls] for o in self.owners],
            [['probe','cheap','cheap_rescue','ocr','select','visual_read'],['probe','cheap','ocr','select','visual_read']])
        for owner in self.owners:
            for op,args,options in owner.calls:
                self.assertTrue(0<options['timeout']<=600)
                if op=='visual_read':self.assertEqual(set(args),{'name','regions'})
        self.assertFalse(report['quality_qualified']);self.assertFalse(report['ocr_accuracy_gate'])
        self.assertFalse(report['actual_model_execution_verified']);self.assertFalse(report['target_runtime_accepted'])
        self.assertTrue(report['model_owners_stopped']);self.assertTrue(report['coordinator_lease_released'])
    def test_x_is_optional_and_uses_a_separate_non_generation_owner(self):
        report=self.run_smoke(qwen=['qwen-4b'],include_yolo_x=True)
        self.assertEqual(report['status'],'pass',report.get('error'))
        self.assertEqual(len(self.owners),2);self.assertEqual(report['completed_generation_calls'],3)
        self.assertEqual([op for op,_,_ in self.owners[-1].calls],['probe','cheap_rescue'])
        self.assertEqual(self.owners[-1].settings.rescue_text_detector,'yolo26x')
    def test_detector_geometry_uncertainty_or_no_text_is_coverage_not_a_cuda_failure(self):
        def incomplete(op,result):
            if op not in {'cheap','cheap_rescue'}:return
            result.update(completed=False,observations=[],regions=[])
            result['diagnostics'].update(lines_processed=0,ctc=[])
        report=self.run_smoke(mutate=incomplete,qwen=['qwen-0.8b'])
        self.assertEqual(report['status'],'pass',report.get('error'))
        coverage=[s['detector_coverage'] for s in report['arms'][0]['steps'] if 'detector_coverage' in s]
        self.assertEqual(len(coverage),2)
        self.assertTrue(all(c['quality_or_coverage_incomplete'] and not c['ctc_path_exercised'] for c in coverage))
        self.assertFalse(report['quality_qualified'])
    def test_missing_requested_weights_stop_before_owner_creation(self):
        factory=Mock()
        result=smoke.run_smoke(self.root/'missing',self.root/'run',self.lock_dir,
            _owner_factory=factory,_verify_models=lambda *a:{'ok':False,'errors':['qwen-4b:missing']},
            _sampler_factory=FakeSampler)
        self.assertEqual(result['status'],'fail');self.assertEqual(result['call_attempts'],0)
        factory.assert_not_called();self.assertTrue(result['coordinator_lease_released'])
    def test_first_error_has_no_retry_and_all_remaining_arms_are_unstarted(self):
        result=self.run_smoke(failure=('ocr',RuntimeError('controlled model failure')))
        self.assertEqual(result['status'],'fail');self.assertEqual(result['failed_stage'],'ocr:0')
        self.assertEqual(len(self.owners),1);self.assertTrue(self.owners[0].closed)
        self.assertEqual(result['neural_generation_calls_attempted'],1)
        self.assertEqual(result['completed_generation_calls'],0)
        self.assertEqual(json.loads((self.root/'run/report.json').read_text())['status'],'fail')
    def test_deadline_or_cancel_preserves_durable_counts_and_closes_lease(self):
        for index,error in enumerate([ModelDeadline('test deadline'),ModelCanceled('test cancel')]):
            report=self.run_smoke(failure=('ocr',error),folder='run'+str(index))
            self.assertEqual(report['status'],'budget_exhausted' if index==0 else 'interrupted')
            self.assertEqual(report['neural_generation_calls_attempted'],1)
            self.assertTrue(report['model_owners_stopped']);self.assertTrue(report['coordinator_lease_released'])
    def test_cancellation_between_completed_calls_starts_no_next_generation(self):
        state={'cancel':False}
        def requested(op):
            if op=='ocr':state['cancel']=True
        report=self.run_smoke(on_call=requested,_cancel=lambda:state['cancel'])
        self.assertEqual(report['status'],'interrupted')
        self.assertEqual(report['completed_generation_calls'],1)
        self.assertEqual(self.owners[0].calls[-1][0],'ocr')
        self.assertEqual(len(self.owners),1);self.assertTrue(self.owners[0].closed)
    def test_measurement_cleanup_failure_cannot_mask_original_model_error(self):
        class BrokenResult(FakeSampler):
            def result(self):raise ValueError('secondary measurement failure')
        owners=[]
        def factory(settings):
            owner=FakeOwner(settings,self.specs,failure=('ocr',RuntimeError('original model failure')))
            owners.append(owner);return owner
        report=smoke.run_smoke(self.root/'models',self.root/'cleanup',self.lock_dir,
            _owner_factory=factory,_verify_models=lambda *a:{'ok':True},_sampler_factory=BrokenResult)
        self.assertEqual(report['status'],'fail');self.assertEqual(report['error']['type'],'RuntimeError')
        self.assertEqual(report['error']['message'],'original model failure')
        self.assertEqual(report['arms'][0]['resource_error_type'],'ValueError')
        self.assertTrue(owners[0].closed);self.assertTrue(report['coordinator_lease_released'])
    def test_cpu_parameter_or_input_and_unknown_metadata_never_pass(self):
        mutations=[lambda r:r.update(device='cpu'),lambda r:r.update(effective_parameter_devices=['cpu']),
                   lambda r:r.update(input_tensor_devices={'input_ids':'cpu'}),
                   lambda r:r.pop('input_tensor_devices'),lambda r:r.update(effective_parameter_dtypes=['torch.float32']),
                   lambda r:r.update(stop_reason='max_new_tokens')]
        for index,mutation in enumerate(mutations):
            def mutate(op,result):
                if op=='ocr':mutation(result)
            report=self.run_smoke(mutate=mutate,folder='bad'+str(index),qwen=['qwen-0.8b'])
            self.assertEqual(report['status'],'fail');self.assertFalse(report['target_runtime_accepted'])
            self.assertEqual(report['completed_generation_calls'],0)
    def test_wrong_ocr_source_or_visual_provenance_fails_before_accepting_generation(self):
        for index,operation in enumerate(('ocr','visual_read')):
            def mutate(op,result):
                if op!=operation:return
                if op=='ocr':result['input_image_sha256']='f'*64
                else:result['provenance'][0]['source_sha256']='f'*64
            report=self.run_smoke(mutate=mutate,folder='provenance'+str(index),qwen=['qwen-0.8b'])
            self.assertEqual(report['status'],'fail');self.assertFalse(report['target_runtime_accepted'])
    def test_busy_coordinator_blocks_before_verification_without_replacing_inode(self):
        target=self.lock_dir/'gpu-delivery.lock';before=target.stat().st_ino
        with FileLease(target),patch('harness_ocr.models.verify_models') as verify:
            report=smoke.run_smoke(self.root/'models',self.root/'blocked',self.lock_dir,
                                   _owner_factory=Mock(),_sampler_factory=FakeSampler)
        self.assertEqual(report['status'],'fail');self.assertEqual(report['error']['type'],'LeaseBusy')
        self.assertEqual(report['call_attempts'],0);verify.assert_not_called()
        self.assertEqual(target.stat().st_ino,before)
    def test_existing_attempt_and_invalid_budget_are_never_overwritten(self):
        folder=self.root/'prior';folder.mkdir();(folder/'report.json').write_text('keep')
        with self.assertRaises(FileExistsError):smoke.run_smoke(self.root/'models',folder,self.lock_dir)
        self.assertEqual((folder/'report.json').read_text(),'keep')
        for value in (0,-1,True,float('nan'),float('inf'),3601):
            with self.assertRaises(ValueError):smoke.run_smoke(self.root/'models',self.root/'invalid',self.lock_dir,total_seconds=value)
        self.assertFalse((self.root/'invalid').exists())


class DeliveryCliContracts(unittest.TestCase):
    def test_missing_4b_and_yolo_files_have_nonzero_module_exit(self):
        with tempfile.TemporaryDirectory() as models:
            code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('harness_ocr.cli',run_name='__main__')"
            result=subprocess.run([sys.executable,'-X','utf8','-c',code,str(ROOT/'src'),'verify-models',
                '--models',models,'--only','qwen-4b','yolo26l-text'],capture_output=True,text=True,encoding='utf-8',timeout=20)
        self.assertNotEqual(result.returncode,0,'Missing models must not return a successful process receipt')
        self.assertFalse(json.loads(result.stdout)['ok'])
    def test_deployment_budget_environment_roundtrips_and_rejects_invalid_values(self):
        values={'HARNESS_MAX_INPUT_TOKENS':'16384','HARNESS_MAX_NEW_TOKENS':'512',
                'HARNESS_CALL_TIMEOUT_SECONDS':'75.5','HARNESS_DOCUMENT_TIMEOUT_SECONDS':'400'}
        with patch.dict(os.environ,values):settings=Settings.from_env();settings.validate()
        self.assertEqual((settings.max_input_tokens,settings.max_new_tokens),(16384,512))
        self.assertEqual((settings.call_timeout_seconds,settings.document_timeout_seconds),(75.5,400))
        for key,value in [('HARNESS_MAX_INPUT_TOKENS','0'),('HARNESS_MAX_NEW_TOKENS','-1'),
                          ('HARNESS_CALL_TIMEOUT_SECONDS','nan'),('HARNESS_DOCUMENT_TIMEOUT_SECONDS','0')]:
            with self.subTest(key=key),patch.dict(os.environ,{**values,key:value}),self.assertRaises(ValueError):
                Settings.from_env().validate()
    def test_new_component_entrypoints_help_do_not_import_torch_or_run_models(self):
        for name in ('gpu_smoke','init_shared_locks'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/(name+'.py')),'--help'],
                                  capture_output=True,text=True,encoding='utf-8',timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('usage:',result.stdout.lower())


@unittest.skipUnless(os.name=='posix','Actual POSIX permission/inode tests run in Linux CI')
class SharedLockInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'locks';self.folder.mkdir()
    def test_fixed_five_modes_preserve_inode_contents_and_live_lease(self):
        for name in locks.FILES:(self.folder/name).write_bytes(b'existing lease bytes')
        before={p.name:(p.stat().st_ino,p.read_bytes()) for p in self.folder.iterdir()}
        target=self.folder/'cuda-admission.lock'
        with FileLease(target):
            report=locks.initialize_locks(self.folder,_test_directory=True)
            with self.assertRaises(LeaseBusy):FileLease(target).acquire()
        self.assertEqual(report['status'],'pass');self.assertFalse(report['production_initialized'])
        self.assertEqual(stat.S_IMODE(self.folder.stat().st_mode),0o1777)
        self.assertEqual(set(p.name for p in self.folder.iterdir()),set(locks.FILES))
        for p in self.folder.iterdir():
            self.assertEqual((p.stat().st_ino,p.read_bytes()),before[p.name])
            self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o666)
    def test_symlink_fifo_and_hardlink_targets_are_rejected_without_changing_outside_file(self):
        outside=self.root/'outside';outside.write_bytes(b'preserve')
        target=self.folder/locks.FILES[0]
        for kind in ('symlink','fifo','hardlink'):
            with self.subTest(kind=kind):
                if kind=='symlink':target.symlink_to(outside)
                elif kind=='fifo':os.mkfifo(target)
                else:os.link(outside,target)
                with self.assertRaises((ValueError,OSError)):
                    locks.initialize_locks(self.folder,_test_directory=True)
                self.assertEqual(outside.read_bytes(),b'preserve')
                target.unlink()
    def test_production_entrypoint_refuses_any_other_directory(self):
        with self.assertRaises(ValueError):locks.initialize_locks(self.folder)
        self.assertFalse(list(self.folder.iterdir()))
    def test_symlink_directory_is_rejected_before_touching_its_target(self):
        outside=self.root/'outside-directory';outside.mkdir(mode=0o700)
        link=self.root/'linked-locks';link.symlink_to(outside,target_is_directory=True)
        before=outside.stat().st_mode
        with self.assertRaises(ValueError):locks.initialize_locks(link,_test_directory=True)
        self.assertEqual(outside.stat().st_mode,before);self.assertFalse(list(outside.iterdir()))


if __name__=='__main__':unittest.main()
