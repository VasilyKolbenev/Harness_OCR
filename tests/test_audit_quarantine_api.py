"""F06 reviewer outcome -> atomic quarantine -> F08 runtime gate, without models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from harness_ocr.api import create_app
from harness_ocr.service import Service
from harness_ocr.settings import Settings
from harness_ocr.control import AuditController,AuditSampler,AuditLog,DriftMonitor,DocumentRiskCalibrator
from harness_ocr.control.common import connect,digest
from harness_ocr.control.runtime import BindingPolicyGateway,SCORE_CONTRACT_ID
import harness_ocr.control.audit as audit_module

API_TOKEN='a'*32
REVIEW_TOKEN='r'*32

class NoModels:
    def close(self):pass
    def call(self,*args,**kwargs):raise AssertionError('No inference in this fixture')

def calibration_records(prefix,split,n):
    return [{'document_id':f'{prefix}{i}','group_id':f'{prefix}-group-{i}','split':split,
             'risk_score':0.0,'critical_error':False} for i in range(n)]

class AuditQuarantineApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.service=Service(Settings(self.root/'state',self.root/'models',device='cpu',dtype='float32',allow_fixture=True),NoModels(),fixture=True)
        self.engine=self.service.engine
        self.engine.audit=AuditController(AuditSampler(1.0,'private-audit-test-seed'),self.engine.audit_log)
        artifact=self.service.artifacts.put(b'synthetic audit fixture')
        request={'artifact_id':artifact['artifact_id'],'profile_id':'D01','subset':['passport.number']}
        self.job=self.service.jobs.create(request)
        contract=self.engine.compile_contract('D01',subset=request['subset'])
        self.binding=self.engine.binding(contract,self.service.settings.qwen)
        self.engine.documents.create_document(self.job['id'],artifact['source_sha256'],contract['profile_id'],
            metadata={'binding':self.binding,'subset':contract['requested_fields']})
        self.snapshot=self.engine.documents.revise(self.job['id'],observations=[{
            'id':'source1','text':'Номер паспорта: 123456','source_sha256':artifact['source_sha256'],
            'evidence_kind':'image_region','page':1,'page_id':'p1','source_size':[500,100],
            'bbox':[0,0,500,100],'completed':True}],claims=[{
            'id':'claim1','field_key':'passport.number','raw_value':'123456','evidence_ids':['source1'],
            'origin':'rule','party_role':'document','depends_on':[]}])
        self.service.jobs.next()
        self.service.jobs.finish(self.job['id'],{'document_id':self.job['id'],'terminal_status':'completed',
            'completed':True,'decision':'review','fields':{},'revision':self.snapshot['revision'],'reason_codes':[]})
        # Intentionally permissive TEST target. This is lifecycle evidence, not
        # real corpus calibration or a claim of production OCR accuracy.
        calibrator=DocumentRiskCalibrator(target_error_rate=.2,confidence=.95,
            min_calibration_groups=10,min_qualification_groups=20,minimum_coverage=.8,
            required_fields=['passport.number'],score_contract_id=SCORE_CONTRACT_ID)
        self.policy=calibrator.fit(calibration_records('cal','calibration',20)+
             calibration_records('qual','qualification',30),self.binding)
        policies=self.engine.policies
        policies.save_candidate(self.policy,actor_id='fixture-researcher',actor_scopes=['policy:propose'])
        policies.approve(self.policy['policy_id'],actor_id='fixture-reviewer',actor_scopes=['policy:approve'])
        policies.activate(self.policy['policy_id'],actor_id='fixture-operator',actor_scopes=['policy:activate'])
        key=digest(self.binding)
        self.engine.gateway=BindingPolicyGateway({'qualified_bindings':[key],
            'policy_ids':{key:self.policy['policy_id']},'score_contract_id':SCORE_CONTRACT_ID},policies)
        self.client=TestClient(create_app(self.service,API_TOKEN,REVIEW_TOKEN));self.client.__enter__()
        accepted=self.client.get(f'/v1/jobs/{self.job["id"]}/result',headers=self.headers()).json()
        self.assertEqual(accepted['decision'],'accept_for_autofill',accepted)
        self.task=accepted['audit']['task_id'];self.assertTrue(accepted['audit']['selected'])
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    @staticmethod
    def headers(review=False):return {'Authorization':'Bearer '+(REVIEW_TOKEN if review else API_TOKEN)}
    def post(self,payload,review=True,task=None):
        return self.client.post(f'/v1/audits/{task or self.task}/outcome',headers=self.headers(review),json=payload)
    def test_confirmed_error_immediately_quarantines_active_policy_and_is_idempotent(self):
        payload={'critical_error':True,'evidence_ids':['source1']}
        denied=self.post(payload,review=False);self.assertEqual(denied.status_code,403)
        self.assertFalse(self.engine.audit_log.events('audit_outcome'))
        unknown=self.post({**payload,'evidence_ids':['foreign-source']});self.assertEqual(unknown.status_code,422)
        self.assertFalse(self.engine.audit_log.events('audit_outcome'))
        response=self.post(payload);self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(response.json()['becomes_training_gold'])
        with connect(self.engine.control_db) as con:
            row=con.execute('SELECT active FROM control_quarantines WHERE binding_sha256=?',(digest(self.binding),)).fetchone()
        self.assertTrue(row and row['active'])
        # The DB safety action already happened; no Engine.project polling was
        # needed to quarantine the otherwise active qualified policy.
        self.assertEqual(self.engine.policies.get(self.policy['policy_id'])['state'],'active')
        current=self.client.get(f'/v1/jobs/{self.job["id"]}/result',headers=self.headers()).json()
        self.assertEqual(current['decision'],'review')
        self.assertIn('binding_quarantined',current['reason_codes'])
        self.assertTrue(all(f['decision']=='review' for f in current['contract_result']['fields'].values()))
        repeated=self.post(payload);self.assertEqual(repeated.status_code,200,repeated.text)
        self.assertEqual(repeated.json(),response.json())
        self.assertEqual(len(self.engine.audit_log.events('audit_outcome')),1)
        self.assertEqual(len(self.engine.audit_log.events('audit_quarantine')),1)
        conflict=self.post({**payload,'critical_error':False});self.assertEqual(conflict.status_code,422)
    def test_outcomes_reject_unknown_inactive_incomplete_sources_and_forged_authority(self):
        base={'critical_error':False,'evidence_ids':['source1']}
        with self.assertRaises(PermissionError):self.service.record_audit_outcome(self.task,base)
        for extra in [{'authorized':True},{'actor_id':'model'},{'actor_scopes':['audit:write']}]:
            self.assertEqual(self.post({**base,**extra}).status_code,422)
        for invalid in [1,'true']:
            self.assertEqual(self.post({**base,'critical_error':invalid}).status_code,422)
        for revision in [self.snapshot['revision'],999,True]:
            self.assertEqual(self.post({**base,'correction_revision':revision}).status_code,422)
        self.assertEqual(self.post(base,task='audit_'+'0'*32).status_code,404)
        skipped=self.engine.audit.plan(self.job['id'],self.snapshot['revision'],self.binding,
            accepted=False,acceptance_context='fixture-unaccepted')
        self.assertEqual(self.post(base,task=skipped['task_id']).status_code,422)
        original=self.snapshot['observations']['source1']
        incomplete={k:v for k,v in original.items() if k not in {'active','created_revision'}}
        incomplete.update(id='partial',completed=False)
        self.engine.documents.revise(self.job['id'],observations=[incomplete])
        self.assertEqual(self.post({**base,'evidence_ids':['partial']}).status_code,422)
        self.engine.documents.revise(self.job['id'],invalidate=['source1'])
        self.assertEqual(self.post(base).status_code,422)
        self.assertFalse(self.engine.audit_log.events('audit_outcome'))
    def test_correction_revision_must_reference_real_authorized_human_event(self):
        corrected=self.engine.documents.human_correct(self.job['id'],'passport.number','123457',['source1'],
             actor='fixture-human-reviewer',expected_revision=self.snapshot['revision'],authorized=True,party_role='document')
        response=self.post({'critical_error':True,'evidence_ids':['source1'],'correction_revision':corrected['revision']})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['correction_revision'],corrected['revision'])


class AtomicAuditQuarantineTests(unittest.TestCase):
    def test_outcome_and_quarantine_rollback_together_and_replay_does_not_reverse_clearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=AuditLog(Path(tmp)/'control.sqlite')
            control=AuditController(AuditSampler(1,'private'),log)
            binding={'profile_id':'fixture','profile_version':'1','subset_contract_id':'test',
                     'model_bundle_sha256':'a'*64,'policy_version':'1','input_scope_version':'test'}
            task=control.plan('doc',1,binding)
            original=audit_module.append_event
            def failure(con,event_id,event_type,payload):
                if event_type=='audit_quarantine':raise RuntimeError('synthetic crash before quarantine commit')
                return original(con,event_id,event_type,payload)
            args={'critical_error':True,'actor_id':'reviewer','actor_scopes':['audit:write'],'evidence_ids':['source']}
            with patch.object(audit_module,'append_event',side_effect=failure),self.assertRaises(RuntimeError):
                control.record_outcome(task['task_id'],**args)
            self.assertFalse(log.events('audit_outcome'))
            with connect(log.db_path) as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM control_quarantines').fetchone()[0],0)
            monitor=DriftMonitor(log,binding,baseline_acceptance_rate=1.,window_size=1,min_observations=1)
            monitor.observe('prior',binding,accepted=False)
            outcome=control.record_outcome(task['task_id'],**args)
            self.assertTrue(monitor.is_quarantined(binding))
            event=log.events('audit_quarantine')[0]['payload']
            self.assertEqual(event['reason_codes'],['acceptance_rate_drop','confirmed_critical_error'])
            monitor.clear_quarantine(binding,actor_id='operator',actor_scopes=['policy:quarantine:clear'],reason='review_completed')
            self.assertFalse(monitor.is_quarantined(binding))
            self.assertEqual(control.record_outcome(task['task_id'],**args),outcome)
            self.assertFalse(monitor.is_quarantined(binding))
            self.assertEqual(len(log.events('audit_quarantine')),1)
    def test_noncritical_review_is_logged_without_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=AuditLog(Path(tmp)/'control.sqlite');controller=AuditController(AuditSampler(1,'s'),log)
            binding={'profile_id':'fixture','profile_version':'1','subset_contract_id':'test',
                     'model_bundle_sha256':'a'*64,'policy_version':'1','input_scope_version':'test'}
            task=controller.plan('doc',1,binding)
            controller.record_outcome(task['task_id'],critical_error=False,actor_id='reviewer',actor_scopes=['audit:write'],evidence_ids=['source'])
            self.assertEqual(len(log.events('audit_outcome')),1);self.assertFalse(log.events('audit_quarantine'))
            with connect(log.db_path) as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM control_quarantines').fetchone()[0],0)

if __name__=='__main__':unittest.main()
