import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from harness_ocr.settings import Settings
from harness_ocr.benchmark import run_benchmark,score
from harness_ocr.ingest import decode
from harness_ocr.util import file_sha

class FakeBackend:
    def __init__(self,settings):self.settings=settings;self.calls=[]
    def close(self):pass
    def call(self,op,args,cancel=None,timeout=None):
        self.calls.append((op,args))
        if op=='ingest':return decode(args['path'],args['output'],self.settings)
        if op=='cheap':return {'observations':[],'regions':[],'completed':True}
        if op=='ocr':return {'text':'Номер паспорта: 123456','completed':True,'seconds':.1,'generated_tokens':8}
        if op=='select':
            prompt=json.loads(args['prompt'])
            candidates=[c for c in prompt['candidates'] if c['role_anchor_observed']]
            proposals=[]
            for field in prompt['fields']:
                match=[c for c in candidates if c['field_key']==field['key']]
                if match:proposals.append({'field_key':field['key'],'candidate_ids':[match[0]['id']],'party_role':field['role']})
            return {'text':json.dumps({'proposals':proposals,'reread_requests':[]}),
                    'completed':True,'seconds':.1,'generated_tokens':30}
        raise ValueError('unexpected_test_operation:'+op)

class BenchmarkTests(unittest.TestCase):
    def test_paired_frozen_optics_gold_not_in_prompt_and_group_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'doc.png';Image.new('RGB',(200,100),'white').save(image)
            settings=Settings(root/'state',root/'models',device='cpu',dtype='float32')
            backend=FakeBackend(settings)
            manifest={'schema_version':'harness-bench/1','split':'dev','documents':[
              {'id':f'v{i}','original_id':'one-original','path':'doc.png','profile_id':'D01',
               'subset':['passport.number'],'gold':{'passport.number':'999999'}} for i in range(2)]}
            p=root/'manifest.json';p.write_text(json.dumps(manifest),encoding='utf-8')
            out=root/'results';summary=run_benchmark(p,out,settings,include_controller=False,backend=backend)
            self.assertEqual(summary['paired_originals'],1)
            self.assertEqual(summary['completed_rows'],6)
            self.assertTrue(summary['protocol']['fixture'])
            self.assertEqual(len(summary['matched_optics_proofs']),2)
            self.assertTrue(all(proof['match'] for proof in summary['matched_optics_proofs']))
            self.assertEqual(summary['matched_optics_proofs'][0]['source_sha256'],file_sha(image))
            self.assertIsNone(summary['resource_measurement_scope']['host_rss_bytes'])
            self.assertFalse(summary['resource_measurement_scope']['capacity_qualified'])
            self.assertEqual(sum(op=='ocr' for op,_ in backend.calls),2)
            for arm in summary['arms']:
                self.assertEqual(summary['metrics'][arm]['wrong'],2)
                self.assertEqual(summary['metrics'][arm]['correct'],0)
                self.assertEqual(summary['metrics'][arm]['original_groups'],1)
            for op,args in backend.calls:
                if op=='select':self.assertNotIn('999999',args['prompt'])
            for i in range(2):
                rows=[json.loads((out/f'result-{i}-{a}.json').read_text()) for a in summary['arms']]
                self.assertEqual(len({r['initial_optics_sha256'] for r in rows}),1)
            with self.assertRaises(ValueError):run_benchmark(p,out,settings,backend=backend)
    def test_score_raw_is_not_normalized_and_withheld_is_not_wrong(self):
        result={'fields':{'x':{'raw_value':'123 456','extraction_status':'observed'},
                          'y':{'raw_value':None,'extraction_status':'not_observed'}}}
        s=score(result,{'x':'123456','y':'value'})
        self.assertEqual((s['correct'],s['wrong'],s['withheld']),(0,1,1))

class IngestTests(unittest.TestCase):
    def test_xml_external_entity_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'bad.xml'
            p.write_text('<!DOCTYPE x [<!ENTITY a SYSTEM "file:///etc/passwd">]><x>&a;</x>')
            with self.assertRaises(Exception):decode(p,r/'out',Settings(r/'state',r/'models',device='cpu',dtype='float32'))
    def test_pixel_limit_rejects_without_downscale_and_no_fake_text_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'im.png';Image.new('RGB',(100,100)).save(p)
            with self.assertRaisesRegex(ValueError,'pixel_limit'):
                decode(p,r/'out',Settings(r/'state',r/'models',device='cpu',dtype='float32',max_page_pixels=10))
    def test_generic_xml_is_never_signature_or_schema_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'doc.xml';p.write_text('<x><inn>1234567890</inn></x>')
            out=decode(p,r/'out',Settings(r/'state',r/'models',device='cpu',dtype='float32'))
            obs=out['observations'][0]
            self.assertEqual(obs['structural_validation_status'],'not_checked')
            self.assertNotIn('bbox',obs);self.assertFalse(out['signature_verified'])

if __name__=='__main__':unittest.main()
