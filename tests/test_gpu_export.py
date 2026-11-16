"""Return-packet privacy, missingness and quality boundaries; no models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpu_export',ROOT/'scripts/export_gpu_delivery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReturnPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'comparison').mkdir()
    def put(self,name,value):
        p=self.root/name;p.parent.mkdir(exist_ok=True,parents=True)
        p.write_text(json.dumps(value),encoding='utf-8')
    def summary(self):
        return {'schema_version':'harness-gpu-delivery-summary/1','status':'completed',
            'technical_execution_passed':True,'planned_rows':1,'recorded_rows':1,'missing_rows':0,
            'rows':[{'public_document_id':'doc_'+'a'*20,'arm':'qwen08b_db','status':'completed',
                'technical_execution_passed':True,'requested_fields':2,'observed_fields':1,
                'document_resources':{'wall_seconds':1.2,'peak_sampled_attributable_gpu_bytes':None,
                    'measurement_errors':['private details must never leave']}}]}
    def run_export(self):
        module.export(self.root,self.root/'RETURN_TO_TEAM.zip')
        return json.loads((self.root/'summary-public.json').read_text(encoding='utf-8'))
    def test_failed_preflight_still_produces_useful_packet_without_invented_accuracy(self):
        self.put('handoff-status.json',{'status':'failed','reason':'sensitive operator path'})
        p=self.run_export()
        self.assertEqual(p['comparison']['status'],'not_run')
        self.assertFalse(p['gpu_inference_claim']);self.assertFalse(p['comparison']['correctness_measured'])
        with zipfile.ZipFile(self.root/'RETURN_TO_TEAM.zip') as z:
            self.assertEqual(set(z.namelist()),{'report.html','metrics.csv','summary-public.json','manifest.json'})
            self.assertIsNone(z.testzip())
    def test_no_literals_paths_logs_tokens_or_original_ids_in_return_zip(self):
        secret='DO_NOT_EXPORT_PRIVATE_SENTINEL'
        s=self.summary();s.update(input_path=secret,raw_value=secret,environment={secret:secret})
        s['rows'][0].update(original_id=secret,raw_value=secret,prompt=secret,error=secret)
        s['rows'][0]['document_resources']['measurement_errors']=[secret]
        self.put('comparison/summary.json',s)
        self.put('handoff-status.json',{'status':'completed','inputs':[secret], 'token':secret,'image_id':secret})
        self.put('doctor.json',{'status':'fail','checks':[{'check':'candidate_runtime','status':'fail',
            'detail':{'torch':secret,'gpu':secret,'token':secret}}]})
        self.put('smoke/report.json',{'status':'fail','error':{'message':secret},'text':secret})
        self.put('rows/private.json',{'value':secret});self.put('logs/private.json',{'value':secret})
        self.run_export()
        with zipfile.ZipFile(self.root/'RETURN_TO_TEAM.zip') as z:
            for name in z.namelist():self.assertNotIn(secret.encode(),z.read(name),name)
    def test_unknown_vram_and_missing_gold_are_not_zero_or_accuracy(self):
        self.put('handoff-status.json',{'image_id':'sha256:'+'b'*64})
        self.put('comparison/summary.json',self.summary());p=self.run_export();a=p['arm_totals'][0]
        self.assertEqual(p['handoff']['image_id'],'sha256:'+'b'*64)
        self.assertIsNone(a['peak_sampled_gpu_bytes']);self.assertIsNone(a['correct'])
        self.assertEqual(a['requested_fields'],2);self.assertFalse(p['comparison']['correctness_measured'])
    def test_gold_counts_are_exact_partial_coverage_and_never_copy_raw_fields(self):
        s=self.summary();s['quality']={'measured':True,'rows':[{'arm':'qwen08b_db',
            'public_document_id':'doc_'+'a'*20,'score':{'measured':True,'total':1,'correct':1,
                'wrong':0,'withheld':0,'fields':[{'raw_value':'PRIVATE','correct':True,'validation_pass':False}]}}]}
        self.put('comparison/summary.json',s);p=self.run_export();a=p['arm_totals'][0]
        self.assertTrue(p['comparison']['correctness_measured']);self.assertEqual(a['correct'],1)
        self.assertEqual(a['correct_validation_pass'],0);self.assertEqual(a['scored_fields'],1)
        self.assertEqual(a['unscored_fields'],1);self.assertNotIn('PRIVATE',json.dumps(p))
    def test_missing_result_reference_is_unknown_not_model_error(self):
        s=self.summary();s.update(status='budget_exhausted',technical_execution_passed=False,missing_rows=1,recorded_rows=0)
        s['rows'][0].update(status='missing',technical_execution_passed=False,observed_fields=None)
        s['quality']={'measured':True,'rows':[{'arm':'qwen08b_db','public_document_id':'doc_'+'a'*20,
            'score':{'measured':False,'missing_result':True,'planned_fields':2}}]}
        self.put('comparison/summary.json',s);p=self.run_export();a=p['arm_totals'][0]
        self.assertFalse(p['comparison']['correctness_measured']);self.assertEqual(a['unknown_reference_fields'],2)
        self.assertIsNone(a['wrong']);self.assertIsNone(a['withheld'])
    def test_injected_smoke_never_turns_into_actual_gpu_acceptance(self):
        self.put('smoke/report.json',{'status':'pass','target_runtime_accepted':True,
            'actual_model_execution_verified':True,'test_dependencies_injected':True})
        self.put('handoff-status.json',{'status':'completed','gpu_inference_verified':True})
        self.assertFalse(self.run_export()['gpu_inference_claim'])

    def test_all_referenced_results_missing_preserves_unknown_coverage(self):
        s=self.summary();s.update(status='budget_exhausted',technical_execution_passed=False,missing_rows=1,recorded_rows=0)
        s['rows'][0].update(status='missing',technical_execution_passed=False,observed_fields=None)
        s['quality']={'measured':False,'gold_sha256':'c'*64,'rows':[{
            'arm':'qwen08b_db','public_document_id':'doc_'+'a'*20,
            'score':{'measured':False,'missing_result':True,'planned_fields':2}}]}
        self.put('comparison/summary.json',s);p=self.run_export();a=p['arm_totals'][0]
        self.assertTrue(p['comparison']['references_provided']);self.assertFalse(p['comparison']['correctness_measured'])
        self.assertEqual(a['reference_fields_planned'],2);self.assertEqual(a['unknown_reference_fields'],2)
        self.assertIsNone(a['wrong']);self.assertIsNone(a['correct'])

    def test_scoring_deadline_keeps_declared_reference_counts_without_fake_scores(self):
        s=self.summary();s['quality']={'measured':False,'references_provided':True,
            'gold_sha256':'c'*64,'reference_field_count_per_arm':2}
        self.put('comparison/summary.json',s);p=self.run_export();a=p['arm_totals'][0]
        self.assertFalse(p['comparison']['correctness_measured']);self.assertEqual(a['scored_fields'],0)
        self.assertEqual(a['reference_fields_planned'],2);self.assertEqual(a['unknown_reference_fields'],2)
        self.assertEqual(a['unscored_fields'],0);self.assertIsNone(a['wrong'])
    def test_truncated_report_is_explicitly_unknown_and_exports(self):
        (self.root/'comparison/summary.json').write_text('{"broken',encoding='utf-8')
        p=self.run_export();self.assertEqual(p['comparison']['status'],'unknown')
    def test_output_is_create_only_and_false_coverage_is_rejected(self):
        self.put('comparison/summary.json',self.summary());self.run_export()
        with self.assertRaises(ValueError):module.export(self.root,self.root/'RETURN_TO_TEAM.zip')
        s=self.summary();s['planned_rows']=2
        with self.assertRaises(ValueError):module.public_summary(s)


if __name__=='__main__':unittest.main()
