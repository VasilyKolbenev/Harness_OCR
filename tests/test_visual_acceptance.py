"""Target diagnostic contracts with image-backed fake model outputs; no NN."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from harness_ocr.inference.constraints import build_response_schema, ENGINE
from harness_ocr.inference.supervisor import ModelDeadline
from harness_ocr.inference.vision import (VERSION,blind_prompt,parse_visual_output,prepare_visual_regions)
from harness_ocr.util import digest,file_sha

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('target_visual_acceptance',ROOT/'scripts'/'visual_acceptance.py')
acceptance=importlib.util.module_from_spec(spec);spec.loader.exec_module(acceptance)


class FakeOwner:
    def __init__(self,settings,specs,*,failure=None,mutate=None,literal=None):
        self.settings=settings;self.specs=specs;self.failure=failure;self.mutate=mutate;self.literal=literal
        self.calls=[];self.closed=False
    def call(self,op,args,**kwargs):
        self.calls.append((op,copy.deepcopy(args),kwargs))
        # A durable, in-progress report must already exist while the model runs.
        report=json.loads((self.settings.data_dir.parent/'report.json').read_text(encoding='utf-8'))
        if report['status']!='running' or not report['stage'].startswith('actual_blind_visual_generation:'):
            raise AssertionError('early_report_missing')
        if self.failure:raise self.failure
        _,provenance=prepare_visual_regions(args['regions']);name=args['name']
        prompt=blind_prompt([p['region_id'] for p in provenance]);literal=self.literal or acceptance.LITERALS[name][0]
        output={'text':json.dumps({'readings':{'r1':{'status':'readable','literal':literal}}}),
            'completed':True,'stop_reason':'eos','fresh_generation_state':True,'modality':'vision',
            'purpose':'blind_optical_capture','adapter_version':VERSION,'model_id':name,
            'device':self.settings.device,'dtype':self.settings.dtype,'processor_class':'Qwen3VLProcessor',
            'json_constraint_applied':True,'proposed_value_supplied':False,'ground_truth_supplied':False,
            'automatic_acceptance':False,'statistical_independence_claimed':False,
            'effective_parameter_dtypes':['torch.'+self.settings.dtype],
            'model_bundle_sha256':digest(self.specs[name]),'blind_prompt':prompt,'blind_prompt_sha256':digest(prompt),
            'response_schema_sha256':digest(build_response_schema(prompt,'visual_read')),'constraint_engine':ENGINE,
            'provenance':provenance,'input_ids_sha256':digest([1,2,3]),'pixel_values_sha256':digest('fake-pixel-tensor'),
            'input_tokens':144,'generated_tokens':50,'effective_eos_token_ids':[7],
            'image_grid_thw':[[1,8,26]],'pixel_values_shape':[208,1536]}
        output['captures']=parse_visual_output(output,provenance)
        if self.mutate:self.mutate(output)
        return output
    def close(self):self.closed=True


class VisualAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.models=self.root/'models';self.specs={};self.owners=[]
        # These are explicit fixture artifacts, not production model files.
        for name in acceptance.QWENS:
            path=self.models/name;path.mkdir(parents=True);files=[]
            for filename in ('config.json','preprocessor_config.json','tokenizer_config.json','chat_template.jinja'):
                target=path/filename;target.write_text('fixture:'+name+':'+filename,encoding='utf-8')
                files.append({'file':filename,'sha256':file_sha(target),'bytes':target.stat().st_size})
            self.specs[name]={'repo':'fixture/'+name,'revision':'b'*40,'files':files}
        self.patcher=patch('harness_ocr.models.model_lock',return_value={'models':self.specs})
        self.patcher.start();self.addCleanup(self.patcher.stop)
    def run_diagnostic(self,*,failure=None,mutate=None,literal=None,**kwargs):
        def factory(settings):
            owner=FakeOwner(settings,self.specs,failure=failure,mutate=mutate,literal=literal)
            self.owners.append(owner);return owner
        return acceptance.run_visual_acceptance(self.root/'run',self.models,
            _owner_factory=factory,_verify_models=lambda *a:{'ok':True,'errors':[],'checked_files':8},**kwargs)
    def test_two_models_are_sequential_one_generation_per_arm_and_wrong_proposal_is_pure(self):
        report=self.run_diagnostic()
        self.assertEqual(report['status'],'pass');self.assertEqual(report['device'],'cuda')
        self.assertEqual(report['dtype'],'bfloat16');self.assertEqual(report['neural_generation_calls'],2)
        self.assertEqual(len(self.owners),1);owner=self.owners[0];self.assertTrue(owner.closed)
        self.assertEqual([c[1]['name'] for c in owner.calls],list(acceptance.QWENS))
        for arm,(op,args,options) in zip(report['arms'],owner.calls):
            self.assertEqual(op,'visual_read');self.assertEqual(set(args),{'name','regions','max_new_tokens'})
            self.assertTrue(0<options['timeout']<=600);self.assertEqual(arm['generation_calls'],1)
            self.assertEqual(arm['correct_proposal_comparison']['verdict'],'correct')
            self.assertEqual(arm['wrong_proposal_comparison']['verdict'],'incorrect')
            self.assertEqual(arm['correct_proposal_comparison']['capture_id'],arm['wrong_proposal_comparison']['capture_id'])
            self.assertFalse(arm['second_generation_for_wrong_proposal'])
            for value in acceptance.LITERALS[arm['model_id']]:self.assertNotIn(value,json.dumps(args))
        self.assertFalse(report['quality_qualified']);self.assertFalse(report['automatic_acceptance_authority'])
        self.assertFalse(report['actual_model_execution_verified']);self.assertFalse(report['target_runtime_accepted'])
        self.assertTrue(report['test_dependencies_injected'])
    def test_cpu_is_explicit_float32_and_one_selected_model_means_one_call(self):
        report=self.run_diagnostic(device='cpu',qwen=['qwen-0.8b'])
        self.assertEqual(report['status'],'pass');self.assertEqual(report['dtype'],'float32')
        self.assertEqual(report['neural_generation_calls'],1);self.assertEqual(len(self.owners[0].calls),1)
        self.assertFalse(report['default_both_models_accepted'])
    def test_missing_model_blocks_before_any_model_owner_is_created(self):
        factory=Mock(side_effect=AssertionError('must not construct a model owner'))
        report=acceptance.run_visual_acceptance(self.root/'run',self.models,_owner_factory=factory,
            _verify_models=lambda *a:{'ok':False,'errors':['qwen-4b/model:missing'],'checked_files':0})
        self.assertEqual(report['status'],'fail');self.assertEqual(report['neural_generation_calls'],0)
        self.assertFalse(report['target_runtime_accepted']);factory.assert_not_called()
        self.assertEqual(json.loads((self.root/'run'/'report.json').read_text())['status'],'fail')
    def test_deadline_stops_remaining_arms_preserves_report_and_closes_owner(self):
        report=self.run_diagnostic(failure=ModelDeadline('controlled deadline'))
        self.assertEqual(report['status'],'fail');self.assertEqual(report['error']['type'],'ModelDeadline')
        self.assertEqual(report['neural_generation_calls'],1);self.assertEqual(len(report['arms']),1)
        self.assertTrue(self.owners[0].closed);self.assertTrue(report['model_owner_stopped'])
        self.assertFalse(report['target_runtime_accepted'])
    def test_backend_failure_does_not_retry_or_false_pass(self):
        report=self.run_diagnostic(failure=RuntimeError('controlled failure'))
        self.assertEqual(report['status'],'fail');self.assertEqual(len(self.owners[0].calls),1)
        self.assertTrue(self.owners[0].closed);self.assertFalse(report['target_runtime_accepted'])
    def test_completed_valid_json_with_wrong_literal_cannot_pass(self):
        report=self.run_diagnostic(literal='999999')
        self.assertEqual(report['status'],'fail');self.assertEqual(report['error']['message'],'constructed_literal_selftest_failed')
        self.assertEqual(report['arms'][0]['correct_proposal_comparison']['verdict'],'incorrect')
        self.assertEqual(report['neural_generation_calls'],1);self.assertFalse(report['target_runtime_accepted'])
    def test_non_eos_or_old_generation_or_wrong_dtype_cannot_pass(self):
        mutations=[lambda r:r.update(stop_reason='max_new_tokens'),lambda r:r.update(fresh_generation_state=False),
            lambda r:r.update(effective_parameter_dtypes=['torch.float32']),lambda r:r.update(automatic_acceptance=True)]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                # Separate new directories preserve previous failed reports.
                folder=self.root/f'run-{index}'
                def factory(settings):return FakeOwner(settings,self.specs,mutate=mutate)
                report=acceptance.run_visual_acceptance(folder,self.models,_owner_factory=factory,
                    _verify_models=lambda *a:{'ok':True})
                self.assertEqual(report['status'],'fail');self.assertFalse(report['target_runtime_accepted'])
                self.assertEqual(report['neural_generation_calls'],1)
    def test_changed_processor_bytes_are_rejected_before_generation(self):
        (self.models/'qwen-4b'/'preprocessor_config.json').write_text('tampered')
        report=self.run_diagnostic()
        self.assertEqual(report['status'],'fail');self.assertEqual(report['neural_generation_calls'],0)
        self.assertFalse(self.owners);self.assertIn('pinned_processor_file_sha_mismatch',report['error']['message'])
    def test_invalid_limits_and_existing_output_never_overwrite(self):
        for seconds in [float('inf'),float('nan'),0,True,3601]:
            with self.assertRaises(ValueError):
                acceptance.run_visual_acceptance(self.root/'not-created',self.models,seconds=seconds)
        self.assertFalse((self.root/'not-created').exists())
        path=self.root/'existing';path.mkdir();(path/'report.json').write_text('preserve')
        with self.assertRaises(FileExistsError):acceptance.run_visual_acceptance(path,self.models)
        self.assertEqual((path/'report.json').read_text(),'preserve')


if __name__=='__main__':unittest.main()
