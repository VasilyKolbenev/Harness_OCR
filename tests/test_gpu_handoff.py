"""Host orchestration contracts with real fake-docker subprocesses; no models."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import subprocess
import unittest
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpu_handoff_host',ROOT/'scripts/gpu_handoff.py')
handoff=importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)

FAKE_DOCKER=r'''
import json,os,pathlib,sys,time
a=sys.argv[1:];control=pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
phase='other'; service=None
if a[0]=='version':phase='docker-version'
elif a[:2]==['compose','version']:phase='compose-version'
elif a[0]=='ps':phase='inventory'
elif a[0]=='inspect':phase='inspect-owned' if a[-1].startswith(('controlled-owned-run-','harness-gpu-')) or a[-1]=='c'*64 else 'inspect'
elif a[:2]==['image','inspect']:phase='image-inspect'
elif a[:2]==['volume','create']:phase='volume'
elif a[0]=='stop':phase='stop-owned'
elif 'build' in a:phase='build'
elif 'run' in a:
    name=a[a.index('--name')+1];service=a[a.index('--name')+2]
    if service=='lock-init':phase='lock-init'
    elif 'experiments.gpu_delivery.run' in a:phase='comparison' if '--execute' in a else 'plan'
    elif 'prepare-models' in a:phase='prepare'
    elif 'verify-models' in a:phase='verify-models'
    elif '/app/scripts/doctor.py' in a:phase='doctor'
    elif '/app/scripts/gpu_smoke.py' in a:phase='smoke'
entry={'args':a,'phase':phase,'service':service,'uid':os.getenv('HARNESS_RUN_UID'),
       'gid':os.getenv('HARNESS_RUN_GID'),'image':os.getenv('HARNESS_HANDOFF_IMAGE'),
       'tokens_inherited':bool(os.getenv('HARNESS_API_TOKEN') or os.getenv('HARNESS_REVIEW_TOKEN'))}
with (control/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry)+'\n')
if os.getenv('FAKE_FAIL_STAGE')==phase:print('controlled stage failure');sys.exit(7)
if os.getenv('FAKE_SLEEP_STAGE')==phase:time.sleep(3)
if phase=='docker-version':print('28.0.0')
elif phase=='compose-version':print('2.35.0')
elif phase=='inventory':print('abc123def456' if os.getenv('FAKE_SERVICE') else '')
elif phase=='inspect':
    mode=os.getenv('FAKE_SERVICE');old=mode=='old'
    print(json.dumps([{'Config':{'Labels':{'com.docker.compose.project':'harness-ocr','com.docker.compose.service':'ocr'},
        'Env':['HARNESS_LOCK_DIR='+('/var/lib/harness/locks' if old else '/locks'),'HARNESS_API_TOKEN=DO_NOT_COPY_INSPECT_SECRET']},
        'Mounts':[] if old else [{'Destination':'/locks','Type':'volume','Name':'harness-ocr-device-locks'}]}]))
elif phase=='inspect-owned':
    marker=control/'owned-name.txt'
    if a[-1]!='c'*64:marker.write_text(a[-1])
    name=marker.read_text();stage=name.rsplit('-',1)[-1]
    print(json.dumps([{'Id':'c'*64,'Name':'/'+name,'Config':{'Labels':{
        'com.docker.compose.project':name.rsplit('-',1)[0],
        'com.docker.compose.service':'measured' if stage in {'smoke','comparison'} else 'handoff'}},
        'State':{'Running':not (control/'owned-stopped.txt').exists()}}]))
elif phase=='stop-owned':(control/'owned-stopped.txt').write_text('stopped')
elif phase=='volume':print(a[-1])
elif phase=='image-inspect':print('sha256:'+'a'*64)
elif phase in {'plan','doctor','smoke','comparison'}:
    root=pathlib.Path(os.environ['HARNESS_REPORTS_HOST'])
    if phase=='plan':target=root/'plan'/'plan.json';value={'model_verification_performed':False}
    elif phase=='doctor':target=root/'doctor.json';value={'schema_version':'harness-doctor/1',
        'model_generation_performed':False,'status':'fail' if os.getenv('FAKE_BAD_DOCTOR') else 'pass'}
    elif phase=='smoke':
        target=root/'smoke'/'report.json';value={'schema_version':'harness-gpu-functional-smoke/1',
            'status':'fail' if os.getenv('FAKE_BAD_SMOKE') else 'pass','quality_qualified':False,
            'actual_model_execution_verified':not bool(os.getenv('FAKE_INJECTED_SMOKE')),
            'target_runtime_accepted':not bool(os.getenv('FAKE_INJECTED_SMOKE')),
            'model_owners_stopped':True,'test_dependencies_injected':bool(os.getenv('FAKE_INJECTED_SMOKE'))}
    else:
        target=root/'comparison'/'summary.json';value={'schema_version':'harness-gpu-delivery-summary/1',
            'technical_execution_passed':not bool(os.getenv('FAKE_BAD_COMPARISON')),'quality_qualified':False,
            'needs_review':True,'accuracy_measured':False}
        q=a[a.index('--qwen')+1];prefixes=['qwen08b','qwen4b'] if q=='both' else ['qwen08b' if q=='qwen-0.8b' else 'qwen4b']
        detectors=['db','db_yolo26l']+(['db_yolo26x'] if '--include-yolo-x' in a else [])
        rows=[{'public_document_id':'doc_'+'f'*20,'arm':q+'_'+d,'status':'completed',
               'technical_execution_passed':value['technical_execution_passed'],
               'requested_fields':1,'observed_fields':0,'validation_pass_fields':0,'review_fields':1,
               'quality_review_fields':0,'model_generation_calls':0,'full_row_wall_seconds':0}
              for q in prefixes for d in detectors]
        value.update(rows=rows,planned_rows=len(rows),recorded_rows=len(rows),missing_rows=0,input_documents=1)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value),encoding='utf-8')
if phase=='verify-models' and os.getenv('FAKE_MUTATE_SOURCE'):
    pathlib.Path(os.environ['FAKE_MUTATE_SOURCE']).write_bytes(b'changed-after-plan')
if phase=='verify-models' and os.getenv('FAKE_MUTATE_GOLD'):
    pathlib.Path(os.environ['FAKE_MUTATE_GOLD']).write_text('{}',encoding='utf-8')
if phase in {'docker-version','compose-version','inventory','inspect','inspect-owned','volume','image-inspect','stop-owned'}:sys.exit(0)
print('fake '+phase)
'''

FAKE_EXPORTER=r'''
import argparse,json,os,pathlib,zipfile
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
status=json.loads((a.run_dir/'handoff-status.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(a.output,'x') as z:z.writestr('status.json',json.dumps({'status':status['status']}))
with (pathlib.Path(os.environ['FAKE_DOCKER_STATE'])/'exports.txt').open('a') as f:f.write(status['status']+'\n')
'''


class HostHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.docs=self.base/'private docs';self.docs.mkdir()
        self.source=self.docs/'paid $HOME.png';self.source.write_bytes(b'fixture-document-not-model-input')
        self.source.chmod(0o600)
        self.control=self.base/'control';self.control.mkdir()
        self.fake=self.control/'fake_docker.py';self.fake.write_text(FAKE_DOCKER,encoding='utf-8')
        self.exporter=self.control/'fake_exporter.py';self.exporter.write_text(FAKE_EXPORTER,encoding='utf-8')
        self.output=self.base/'new reports'
        self.env=patch.dict(os.environ,{'FAKE_DOCKER_STATE':str(self.control),
            'HARNESS_API_TOKEN':'DO_NOT_COPY_API_SECRET','HARNESS_REVIEW_TOKEN':'DO_NOT_COPY_REVIEW_SECRET'},clear=False)
        self.env.start()
    def tearDown(self):
        self.env.stop();self.temp.cleanup()
    def args(self,*extra):
        return handoff.parser().parse_args(['--input',str(self.docs),'--profile','D07',
            '--output',str(self.output),'--models',str(self.base/'models'),
            '--lock-volume','harness-ocr-device-locks','--service-project','harness-ocr','--gpu-id','0',*extra])
    def run_fake(self,args=None):
        return handoff.run_handoff(args or self.args(),docker_command=(sys.executable,str(self.fake)),
            identity=(1234,5678),state_dir=self.base/'state',export_command=(sys.executable,str(self.exporter)))
    def calls(self):
        p=self.control/'calls.jsonl'
        return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
    def status(self):return json.loads((self.output/'handoff-status.json').read_text(encoding='utf-8'))
    def gold(self,documents=None):
        path=self.base/'private $GOLD references.json'
        path.write_text(json.dumps({'schema_version':'harness-gpu-delivery-gold/1',
            'documents':documents if documents is not None else {'doc-0001':{'payer.name':'PRIVATE_EXACT_REFERENCE'}}}),encoding='utf-8')
        path.chmod(0o600)
        return path
    def test_complete_serial_flow_private_inputs_and_no_credentials(self):
        before=self.source.read_bytes();result=self.run_fake()
        self.assertTrue(result['orchestration_contract_passed']);self.assertFalse(result['quality_qualified'])
        self.assertTrue(result['test_dependencies_injected']);self.assertFalse(result['gpu_inference_verified'])
        self.assertFalse(result['technical_execution_passed'])
        self.assertEqual(result['status'],'completed');self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
        calls=self.calls();phases=[c['phase'] for c in calls]
        self.assertEqual(phases,['docker-version','compose-version','inventory','build','image-inspect','volume','lock-init',
            'plan','prepare','verify-models','doctor','smoke','comparison'])
        self.assertTrue(all(not c['tokens_inherited'] for c in calls))
        self.assertTrue(all(c['uid']=='1234' and c['gid']=='5678' for c in calls))
        self.assertEqual(self.source.read_bytes(),before)
        if os.name=='posix':self.assertEqual(stat.S_IMODE(self.source.stat().st_mode),0o600)
        stages=[c for c in calls if c['phase'] in {'doctor','smoke','comparison','plan'}]
        self.assertEqual([c['service'] for c in stages],['handoff','handoff','measured','measured'])
        for call in calls[phases.index('volume'):]:
            self.assertEqual(call['image'],'sha256:'+'a'*64)
            if 'run' in call['args']:self.assertIn('never',call['args'])
        state=next((self.base/'state').iterdir())
        overlay=json.loads((state/'inputs.compose.json').read_text(encoding='utf-8'))
        mount=overlay['services']['handoff']['volumes'][0]
        self.assertEqual(mount['source'],str(self.source.resolve()).replace('$','$$'))
        self.assertTrue(mount['read_only']);self.assertFalse(mount['bind']['create_host_path'])
        self.assertEqual(overlay['services']['handoff'],overlay['services']['measured'])
        manifest=json.loads((state/'input'/'own-documents.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['documents'][0]['profile_id'],'D07')
        self.assertEqual(manifest['documents'][0]['path'],'documents/0000.png')
        self.assertEqual((state/'input'/'documents'/'0000.png').stat().st_size,0)
        all_reports=''.join(p.read_text(encoding='utf-8',errors='ignore') for p in self.output.rglob('*') if p.is_file())
        for secret in ['DO_NOT_COPY_API_SECRET','DO_NOT_COPY_REVIEW_SECRET']:
            self.assertNotIn(secret,all_reports)
    def test_failure_is_not_retried_and_always_exports_partial_status(self):
        with patch.dict(os.environ,{'FAKE_FAIL_STAGE':'smoke'}):
            with self.assertRaisesRegex(handoff.HandoffError,'stage_failed:smoke'):self.run_fake()
        phases=[c['phase'] for c in self.calls()]
        self.assertEqual(phases.count('smoke'),1);self.assertNotIn('comparison',phases)
        self.assertFalse(self.status()['technical_execution_passed'])
        self.assertFalse(self.status()['retry_performed']);self.assertEqual(self.status()['status'],'failed')
        self.assertTrue(self.status()['cleanup']['confirmed_not_running'])
        self.assertEqual(self.status()['cleanup']['validated_container_id'],'c'*64)
        self.assertEqual(phases.count('stop-owned'),1)
        self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
    def test_zero_exit_with_failed_smoke_report_is_not_a_pass(self):
        with patch.dict(os.environ,{'FAKE_BAD_SMOKE':'1'}):
            with self.assertRaisesRegex(handoff.HandoffError,'gpu_smoke_did_not_pass'):self.run_fake()
        self.assertNotIn('comparison',[c['phase'] for c in self.calls()])
    def test_injected_smoke_cannot_claim_actual_target_acceptance(self):
        with patch.dict(os.environ,{'FAKE_INJECTED_SMOKE':'1'}):
            with self.assertRaisesRegex(handoff.HandoffError,'gpu_smoke_did_not_pass'):self.run_fake()
        self.assertFalse(self.status()['gpu_inference_verified'])
        self.assertNotIn('comparison',[c['phase'] for c in self.calls()])
    def test_review_without_gold_is_technical_completion_not_accuracy(self):
        result=self.run_fake()
        self.assertTrue(result['orchestration_contract_passed']);self.assertFalse(result['quality_qualified'])
        summary=json.loads((self.output/'comparison'/'summary.json').read_text())
        self.assertTrue(summary['needs_review']);self.assertFalse(summary['accuracy_measured'])
    def test_manifest_requires_explicit_contract_and_rejects_gold_before_docker(self):
        manifest=self.docs/'own.json'
        manifest.write_text(json.dumps({'schema_version':handoff.SCHEMA,'documents':[{
            'id':'receipt','path':self.source.name,'profile_id':'D07','gold':{'payer.name':'invented'}}]}))
        args=handoff.parser().parse_args(['--manifest',str(manifest),'--output',str(self.output)])
        with self.assertRaisesRegex(handoff.HandoffError,'manifest_document_shape'):self.run_fake(args)
        self.assertFalse(self.calls());self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
    def test_source_escape_and_duplicate_ids_are_rejected(self):
        manifest=self.docs/'own.json'
        for paths in [[{'id':'d1','path':'../outside.png','profile_id':'D07'}],
                      [{'id':'Same','path':self.source.name,'profile_id':'D07'},
                       {'id':'same','path':self.source.name,'profile_id':'D07'}]]:
            manifest.write_text(json.dumps({'schema_version':handoff.SCHEMA,'documents':paths}))
            args=handoff.parser().parse_args(['--manifest',str(manifest),'--output',str(self.output)])
            with self.assertRaises(handoff.HandoffError):handoff.documents_from_args(args)
    def test_old_active_service_is_refused_without_stop_or_inspect_secret(self):
        with patch.dict(os.environ,{'FAKE_SERVICE':'old'}):
            with self.assertRaisesRegex(handoff.HandoffError,'old_harness_lock_namespace'):self.run_fake()
        phases=[c['phase'] for c in self.calls()]
        self.assertNotIn('build',phases);self.assertNotIn('stop-owned',phases)
        self.assertIn('stop ocr',self.status()['reason'])
        self.assertNotIn('DO_NOT_COPY_INSPECT_SECRET',(self.output/'handoff-status.json').read_text())
    def test_active_shared_namespace_service_also_requires_manual_stop(self):
        with patch.dict(os.environ,{'FAKE_SERVICE':'new'}):
            with self.assertRaisesRegex(handoff.HandoffError,'running_harness_service'):self.run_fake()
        self.assertNotIn('stop-owned',[c['phase'] for c in self.calls()])
    def test_prepare_only_is_safe_cache_preparation_not_inference(self):
        result=self.run_fake(self.args('--prepare-only','--include-yolo-x'))
        self.assertEqual(result['status'],'prepared');self.assertFalse(result['technical_execution_passed'])
        self.assertFalse(result['model_inference_performed'])
        calls=self.calls();self.assertNotIn('doctor',[c['phase'] for c in calls])
        self.assertNotIn('smoke',[c['phase'] for c in calls])
        prepare=next(c for c in calls if c['phase']=='prepare')['args']
        for key in ['xiaomi','detector','cyrillic','qwen-0.8b','qwen-4b','yolo26l-text','yolo26x-text']:
            self.assertIn(key,prepare)
    def test_explicit_small_qwen_does_not_download_or_verify_4b(self):
        self.run_fake(self.args('--qwen','qwen-0.8b'))
        calls=self.calls()
        for phase in ('prepare','verify-models','doctor'):
            args=next(c for c in calls if c['phase']==phase)['args']
            self.assertIn('qwen-0.8b',args);self.assertNotIn('qwen-4b',args)
        for phase in ('plan','smoke','comparison'):
            args=next(c for c in calls if c['phase']==phase)['args']
            self.assertEqual(args[args.index('--qwen')+1],'qwen-0.8b')
    def test_explicit_4b_does_not_download_or_verify_small_qwen(self):
        self.run_fake(self.args('--qwen','qwen-4b'))
        calls=self.calls()
        for phase in ('prepare','verify-models','doctor'):
            args=next(c for c in calls if c['phase']==phase)['args']
            self.assertIn('qwen-4b',args);self.assertNotIn('qwen-0.8b',args)
    def test_separate_gold_is_parent_only_readonly_and_never_in_input_manifest(self):
        gold=self.gold();before=gold.read_bytes()
        result=self.run_fake(self.args('--gold',str(gold)))
        self.assertTrue(result['gold_provided']);self.assertEqual(result['gold_reference_fields'],1)
        self.assertFalse(result['reference_values_sent_to_workers'])
        calls=self.calls()
        for c in calls:
            if c['phase'] in {'plan','comparison'}:
                self.assertIn('--gold',c['args'])
                self.assertEqual(c['args'][c['args'].index('--gold')+1],'/references/gold.json')
                refs=[c['args'][i+1] for i,arg in enumerate(c['args']) if arg=='--file'][-1]
                self.assertTrue(refs.endswith('references.compose.json'))
            elif c['phase'] in {'prepare','verify-models','doctor','smoke'}:
                self.assertNotIn('--gold',c['args'])
                self.assertFalse(any(arg.endswith('references.compose.json') for arg in c['args']))
        state=next((self.base/'state').iterdir())
        overlay=json.loads((state/'references.compose.json').read_text(encoding='utf-8'))
        for service in ('handoff','measured'):
            mounts=overlay['services'][service]['volumes']
            source=next(m for m in mounts if m['target']=='/references/gold.json')
            self.assertEqual(source['source'],str(gold.resolve()).replace('$','$$'))
            self.assertTrue(source['read_only']);self.assertFalse(source['bind']['create_host_path'])
        manifest=(state/'input'/'own-documents.json').read_text(encoding='utf-8')
        self.assertNotIn('PRIVATE_EXACT_REFERENCE',manifest);self.assertNotIn('gold',manifest)
        self.assertEqual(gold.read_bytes(),before)
        for path in self.output.rglob('*'):
            if path.is_file():self.assertNotIn(b'PRIVATE_EXACT_REFERENCE',path.read_bytes())
    def test_malformed_and_typed_gold_rejected_before_docker_or_download(self):
        for index,value in enumerate(['bad-json',json.dumps({'schema_version':'wrong','documents':{}}),
                      json.dumps({'schema_version':'harness-gpu-delivery-gold/1','documents':{
                          'doc-0001':{'payer.name':{'schema_version':'typed-fields-v1','items':[]}}}})]):
            gold=self.base/'bad-gold.json';gold.write_text(value)
            self.output=self.base/f'bad-gold-output-{index}'
            with self.assertRaises(handoff.HandoffError):self.run_fake(self.args('--gold',str(gold)))
            self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
        self.assertFalse(self.calls())
    def test_unknown_gold_document_and_field_outside_explicit_subset_fail_preflight(self):
        for index,docs in enumerate(({'unknown-doc':{'payer.name':'reference'}},
                                     {'doc-0001':{'not.requested':'reference'}})):
            gold=self.gold(docs)
            self.output=self.base/f'unknown-gold-output-{index}'
            with self.assertRaises(handoff.HandoffError):
                self.run_fake(self.args('--subset','payer.name','--gold',str(gold)))
        self.assertFalse(self.calls())
    def test_gold_change_during_download_stops_before_model_calls(self):
        gold=self.gold()
        with patch.dict(os.environ,{'FAKE_MUTATE_GOLD':str(gold)}):
            with self.assertRaisesRegex(handoff.HandoffError,'gold_changed_after_preflight'):
                self.run_fake(self.args('--gold',str(gold)))
        self.assertNotIn('doctor',[c['phase'] for c in self.calls()])
        self.assertNotIn('smoke',[c['phase'] for c in self.calls()])
    def test_auto_id_mapping_is_filename_order_and_explicit_gold_is_excluded(self):
        (self.docs/'A-first.png').write_bytes(b'second-fixture')
        gold=self.docs/'references.json';gold.write_text(json.dumps({
            'schema_version':'harness-gpu-delivery-gold/1','documents':{'doc-0001':{'payer.name':'REF'}}}))
        self.run_fake(self.args('--gold',str(gold),'--prepare-only'))
        mapping=json.loads((self.output/'private-input-map.json').read_text(encoding='utf-8'))
        self.assertEqual([(d['id'],d['relative_source_path']) for d in mapping['documents']],
                         [('doc-0001','A-first.png'),('doc-0002',self.source.name)])
        self.assertTrue(mapping['auto_ids'])
    def test_real_public_export_excludes_separate_references_and_private_map(self):
        exporter=ROOT/'scripts/export_gpu_delivery.py'
        if not exporter.exists():self.skipTest('exporter implementation not available yet')
        gold=self.gold()
        handoff.run_handoff(self.args('--gold',str(gold)),docker_command=(sys.executable,str(self.fake)),
            identity=(1234,5678),state_dir=self.base/'state')
        with zipfile.ZipFile(self.output/'RETURN_TO_TEAM.zip') as archive:
            self.assertFalse(any('private-input-map' in name or 'gold' in name for name in archive.namelist()))
            for name in archive.namelist():
                raw=archive.read(name)
                for secret in (b'PRIVATE_EXACT_REFERENCE',b'DO_NOT_COPY_API_SECRET',b'DO_NOT_COPY_REVIEW_SECRET'):
                    self.assertNotIn(secret,raw)
    def test_skip_download_still_verifies_cache_and_no_output_resume(self):
        self.run_fake(self.args('--skip-download'))
        self.assertNotIn('prepare',[c['phase'] for c in self.calls()])
        self.assertIn('verify-models',[c['phase'] for c in self.calls()])
        before=(self.output/'handoff-status.json').read_bytes()
        with self.assertRaisesRegex(handoff.HandoffError,'output_must_be_new'):self.run_fake()
        self.assertEqual((self.output/'handoff-status.json').read_bytes(),before)
    def test_source_change_after_download_blocks_all_model_calls(self):
        with patch.dict(os.environ,{'FAKE_MUTATE_SOURCE':str(self.source)}):
            with self.assertRaisesRegex(handoff.HandoffError,'source_changed_after_preflight'):self.run_fake()
        phases=[c['phase'] for c in self.calls()]
        self.assertNotIn('doctor',phases);self.assertNotIn('smoke',phases)
    def test_invalid_profile_plan_stops_before_download(self):
        with patch.dict(os.environ,{'FAKE_FAIL_STAGE':'plan'}):
            with self.assertRaisesRegex(handoff.HandoffError,'stage_failed:plan'):self.run_fake()
        self.assertNotIn('prepare',[c['phase'] for c in self.calls()])
    def test_intentional_root_host_identity_is_preserved_without_account_changes(self):
        result=handoff.run_handoff(self.args(),docker_command=(sys.executable,str(self.fake)),
            identity=(0,0),state_dir=self.base/'state',export_command=(sys.executable,str(self.exporter)))
        self.assertTrue(result['orchestration_contract_passed'])
        self.assertEqual(result['container_uid'],0);self.assertEqual(result['container_gid'],0)
        self.assertTrue(all(c['uid']=='0' and c['gid']=='0' for c in self.calls()))
        self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
    def test_false_completed_comparison_is_not_accepted(self):
        with patch.dict(os.environ,{'FAKE_BAD_COMPARISON':'1'}):
            with self.assertRaisesRegex(handoff.HandoffError,'comparison_not_technically_complete'):self.run_fake()
        self.assertEqual(self.status()['status'],'failed')
        self.assertFalse(self.status()['technical_execution_passed'])
        self.assertTrue((self.output/'RETURN_TO_TEAM.zip').is_file())
    def test_timeout_stops_only_generated_container_and_does_not_retry(self):
        reports=self.base/'direct-stage';(reports/'logs').mkdir(parents=True)
        docker=handoff.Docker((sys.executable,str(self.fake)),os.environ.copy(),reports,
                              [ROOT/'compose.gpu-handoff.yaml'],'controlled-owned-run')
        with patch.dict(docker.environment,{'FAKE_SLEEP_STAGE':'smoke'}):
            with self.assertRaises(subprocess.TimeoutExpired):
                docker.stage('smoke',['python','/app/scripts/gpu_smoke.py'],timeout=.5,service='measured')
        stops=[c['args'] for c in self.calls() if c['phase']=='stop-owned']
        self.assertEqual(stops,[['stop','--time','30','c'*64]])
        self.assertIsNone(docker.current_container)
        self.assertEqual(docker.last_cleanup['scope'],'exact_owned_container_only')
        self.assertTrue(docker.last_cleanup['stopped_by_docker'])
        self.assertTrue(docker.last_cleanup['confirmed_not_running'])
    def test_compose_network_mount_and_pid_boundaries_are_explicit(self):
        text=(ROOT/'compose.gpu-handoff.yaml').read_text(encoding='utf-8')
        self.assertEqual(text.count('pid: host'),1)
        self.assertEqual(text.count('network_mode: bridge'),1)
        self.assertEqual(text.count('network_mode: none'),2)
        self.assertIn('HARNESS_INSTALL_YOLO: "1"',text)
        self.assertIn('gpu-rc6-yolo',text)
        self.assertIn('cap_add: [CHOWN, FOWNER, DAC_OVERRIDE]',text)
        self.assertIn('command: []',text)
        self.assertIn('tmpfs: ["/tmp:size=2147483648,mode=1777"]',text)
        self.assertIn('tmpfs: ["/tmp:size=536870912,mode=1777"]',text)
        self.assertEqual(text.count('HOME: /tmp'),2)
        self.assertIn('HARNESS_ATTENTION_BACKEND: sdpa',text)
        self.assertIn('external: true',text)
        self.assertNotIn('HARNESS_API_TOKEN',text);self.assertNotIn('ports:',text)


class OwnedCleanupTests(unittest.TestCase):
    def docker(self):
        value=handoff.Docker(('docker',),{},Path('.'),[],'expected-project')
        value.current_container='expected-project-smoke';value.current_service='measured'
        return value
    def inspected(self,running=True,project='expected-project'):
        return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':'c'*64,
            'Name':'/expected-project-smoke','Config':{'Labels':{
                'com.docker.compose.project':project,'com.docker.compose.service':'measured'}},
            'State':{'Running':running}}]))
    def test_rm_removal_is_confirmed_by_successful_empty_inventory(self):
        docker=self.docker();calls=[];responses=[self.inspected(),SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1,stdout=''),SimpleNamespace(returncode=0,stdout='')]
        def command(args,**kwargs):calls.append(args);return responses.pop(0)
        with patch.object(handoff.subprocess,'run',side_effect=command):docker.stop_owned_container()
        self.assertTrue(docker.last_cleanup['confirmed_not_running'])
        self.assertEqual(docker.last_cleanup['final_state'],'absent')
        self.assertEqual(docker.last_cleanup['validated_container_id'],'c'*64)
        self.assertIn(['docker','ps','-a','--filter','id='+'c'*64,'--format','{{.ID}}'],calls)
        self.assertFalse(any(c[1]=='kill' for c in calls))
    def test_failed_stop_uses_kill_only_after_same_owner_is_revalidated(self):
        docker=self.docker();calls=[];responses=[self.inspected(),SimpleNamespace(returncode=1),
            self.inspected(),SimpleNamespace(returncode=0),self.inspected(False)]
        def command(args,**kwargs):calls.append(args);return responses.pop(0)
        with patch.object(handoff.subprocess,'run',side_effect=command):docker.stop_owned_container()
        self.assertTrue(docker.last_cleanup['kill_attempted']);self.assertTrue(docker.last_cleanup['confirmed_not_running'])
        self.assertEqual(calls[3],['docker','kill','c'*64])
        self.assertEqual(calls[2],['docker','inspect','--type','container','c'*64])
    def test_name_collision_or_changed_project_is_never_stopped_or_killed(self):
        for changed_after_stop in (False,True):
            docker=self.docker();calls=[]
            responses=([self.inspected(),SimpleNamespace(returncode=1),self.inspected(project='other-project')]
                       if changed_after_stop else [self.inspected(project='other-project')])
            def command(args,**kwargs):calls.append(args);return responses.pop(0)
            with patch.object(handoff.subprocess,'run',side_effect=command):docker.stop_owned_container()
            self.assertFalse(docker.last_cleanup['confirmed_not_running'])
            self.assertEqual(docker.last_cleanup['final_state'],'ownership_mismatch')
            self.assertFalse(any(c[1]=='kill' for c in calls))
            self.assertEqual(sum(c[1]=='stop' for c in calls),int(changed_after_stop))
    def test_daemon_failure_preserves_exact_generated_name_and_validated_id(self):
        docker=self.docker();responses=[self.inspected(),SimpleNamespace(returncode=1),
            OSError('daemon unavailable'),]
        def command(args,**kwargs):
            result=responses.pop(0)
            if isinstance(result,Exception):raise result
            return result
        with patch.object(handoff.subprocess,'run',side_effect=command):docker.stop_owned_container()
        self.assertEqual(docker.last_cleanup['generated_container_name'],'expected-project-smoke')
        self.assertEqual(docker.last_cleanup['validated_container_id'],'c'*64)
        self.assertFalse(docker.last_cleanup['confirmed_not_running'])
        self.assertEqual(docker.last_cleanup['final_state'],'unavailable')


class CrossProjectGpuGuardTests(unittest.TestCase):
    def container(self,devices=('0',),*,project='legacy_other',shared=False,lock_target='/var/lib/harness/locks'):
        return {'Id':'abc123def456','Config':{'Labels':{'com.docker.compose.project':project,
            'com.docker.compose.service':'ocr'},'Env':['HARNESS_LOCK_DIR='+lock_target,
            'HARNESS_API_TOKEN=INSPECT_SECRET_NEVER_PRINT']},
            'HostConfig':{'Privileged':False,'Devices':[],'DeviceRequests':[
                {'Driver':'nvidia','Count':0,'DeviceIDs':list(devices),'Capabilities':[['gpu']]}]},
            'Mounts':[{'Type':'volume','Name':'harness-ocr-device-locks','Destination':lock_target}] if shared else []}
    def check(self,container,selected='0'):
        calls=[]
        class Inventory:
            def raw(self,arguments):
                calls.append(arguments)
                return 'abc123def456\n' if arguments[0]=='ps' else json.dumps([container])
        return Inventory(),calls
    def test_other_project_same_gpu_is_refused_with_actual_project_and_no_autostop(self):
        docker,calls=self.check(self.container())
        with self.assertRaises(handoff.HandoffError) as error:
            handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks','0')
        message=str(error.exception)
        self.assertIn('old_harness_lock_namespace',message)
        self.assertIn('--project-name legacy_other stop ocr',message)
        self.assertIn('docker stop abc123def456',message)
        self.assertNotIn('INSPECT_SECRET',message)
        self.assertEqual([c[0] for c in calls],['ps','inspect'])
    def test_current_api_mount_path_matches_env_and_is_not_misdiagnosed_as_old(self):
        docker,_=self.check(self.container(project='harness_ocr',shared=True))
        with self.assertRaises(handoff.HandoffError) as error:
            handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks','0')
        self.assertIn('running_harness_service',str(error.exception))
        self.assertNotIn('old_harness_lock_namespace',str(error.exception))
    def test_only_proven_same_family_disjoint_ids_are_ignored(self):
        uuid_a='GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
        uuid_b='GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
        for selected,existing in [('0',('1','2')),(uuid_a,(uuid_b,))]:
            with self.subTest(selected=selected):
                docker,calls=self.check(self.container(existing))
                handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks',selected)
                self.assertEqual([c[0] for c in calls],['ps','inspect'])
        for selected,existing in [('0',(uuid_b,)),(uuid_a,('1',)),('0',('1','0')),
                                  (uuid_a,(uuid_a.upper().replace('GPU-','GPU-'),)),
                                  ('0',('all',)),('0',('MIG-device',))]:
            with self.subTest(selected=selected,existing=existing):
                docker,_=self.check(self.container(existing))
                with self.assertRaises(handoff.HandoffError):
                    handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks',selected)
    def test_unknown_all_or_manual_gpu_access_is_refused_across_projects(self):
        for kind in ('missing','all','privileged','manual'):
            with self.subTest(kind=kind):
                container=self.container(('1',))
                if kind=='missing':container.pop('HostConfig')
                elif kind=='all':container['HostConfig']['DeviceRequests'][0].update(Count=-1,DeviceIDs=[])
                elif kind=='privileged':container['HostConfig']['Privileged']=True
                else:container['HostConfig']['Devices']=[{'PathOnHost':'/dev/nvidia0'}]
                docker,_=self.check(container)
                with self.assertRaises(handoff.HandoffError):
                    handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks','0')
    def test_invalid_project_metadata_never_enters_stop_instruction(self):
        container=self.container(project='$(echo INSPECT_SECRET)')
        docker,_=self.check(container)
        with self.assertRaises(handoff.HandoffError) as error:
            handoff.check_old_service(docker,'harness-ocr','harness-ocr-device-locks','0')
        self.assertIn('docker stop abc123def456',str(error.exception))
        self.assertNotIn('INSPECT_SECRET',str(error.exception))


if __name__=='__main__':unittest.main()
