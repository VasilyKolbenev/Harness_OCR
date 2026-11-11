"""Research adapters execute real pipeline stages with fixture model outputs."""
from __future__ import annotations
import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from harness_ocr.core.trace import TraceRecorder
from harness_ocr.engine import Engine
from harness_ocr.profiles import ProfileRegistry
from harness_ocr.settings import Settings
from harness_ocr.research.integration import (load_trace_snapshot,blind_case_from_trace,
    QwenTextVerifier,run_trace_blind_exam,minimize_trace_failure,replay_selector_graph,
    OracleSelectorEvaluator,run_trace_oracle_study)

SHA='a'*64

def observation(ident,text):
    return {'id':ident,'source_sha256':SHA,'text':text,'completed':True,'evidence_kind':'image_region',
         'page':1,'page_id':'p1','source_size':[800,1200],'bbox':[0,0,800,1200],
         'capture_id':ident,'model_id':'fixture-reader'}

class Backend:
    def __init__(self):self.calls=[];self.incomplete=False
    def call(self,op,args,**kwargs):
        self.calls.append((op,copy.deepcopy(args),kwargs));p=json.loads(args['prompt'])
        if args.get('purpose')=='verifier':
            text=p['candidate']['raw_value'];observed='\n'.join(n['text'] for n in p['source_line_graph'])
            output={'verdict':'correct' if text in observed else 'incorrect'}
        else:
            selected={}
            for c in p['candidates']:
                if c['role_anchor_observed']:selected.setdefault(c['field_key'],c)
            output={'proposals':[{'field_key':k,'candidate_ids':[c['id']],'party_role':c['party_role']}
                       for k,c in selected.items()],'reread_requests':[]}
        return {'text':json.dumps(output),'completed':not self.incomplete,'fresh_generation_state':True}

class TraceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.backend=Backend();settings=Settings(self.root/'state',self.root/'models',device='cpu',dtype='float32')
        self.engine=Engine(settings,self.backend)
        self.engine.run('d1','not-read.png',SHA,'D01',subset=['passport.number'],
              shared_optics={'source_sha256':SHA,'completed':True,'regions':[],
               'observations':[observation('main','Номер паспорта: 654321'),observation('noise','Ненужный заголовок')]})
        self.snapshot=load_trace_snapshot(self.root/'state'/'trace.sqlite','d1')
    def test_trace_chain_verified_and_missing_trace_never_created(self):
        self.assertEqual(self.snapshot['schema_version'],'harness-research-trace-snapshot/1')
        self.assertEqual(len(self.snapshot['observations']),2)
        missing=self.root/'not-created.sqlite'
        with self.assertRaises(FileNotFoundError):load_trace_snapshot(missing,'d1')
        self.assertFalse(missing.exists())
        db=sqlite3.connect(self.root/'state'/'trace.sqlite')
        db.execute("UPDATE stage_trace SET payload='{}' WHERE seq=1");db.commit();db.close()
        with self.assertRaises(Exception):load_trace_snapshot(self.root/'state'/'trace.sqlite','d1')
    def test_real_text_verifier_backend_gets_no_gold_or_private_case_ids(self):
        cid=next(c['id'] for c in self.snapshot['recorded_candidates'].values() if c['role_verified'])
        good=blind_case_from_trace(self.snapshot,case_id='private-case-good',original_id='source-one',
               field_key='passport.number',candidate_id=cid,candidate_correct=True,observable={'text':True})
        bad=blind_case_from_trace(self.snapshot,case_id='private-case-bad',original_id='source-one',
               field_key='passport.number',candidate_id=cid,candidate_correct=False,
               observable={'text':True},proposed_raw='123456',origin='injected')
        backend=Backend();report=run_trace_blind_exam([good,bad],backend,trial_path=self.root/'blind.jsonl')
        self.assertEqual(report['metrics']['detected_bad'],1);self.assertEqual(report['metrics']['retained_good'],1)
        for op,args,kwargs in backend.calls:
            self.assertEqual(op,'select');self.assertEqual(args['purpose'],'verifier')
            for forbidden in ['candidate_correct','private-case-good','private-case-bad','observable','gold']:
                self.assertNotIn(forbidden,args['prompt'])
    def test_incomplete_verifier_is_recorded_error_not_success_or_abstention(self):
        cid=next(iter(self.snapshot['recorded_candidates']))
        case=blind_case_from_trace(self.snapshot,case_id='c1',original_id='source-one',field_key='passport.number',
                   candidate_id=cid,candidate_correct=True,observable={'text':True})
        backend=Backend();backend.incomplete=True
        report=run_trace_blind_exam([case],backend)
        self.assertEqual(report['metrics']['verifier_errors'],1)
        self.assertEqual(report['metrics']['retained_good'],0)
    def test_visual_modality_rejected_before_model_not_silently_text(self):
        backend=Backend();verifier=QwenTextVerifier(backend)
        with self.assertRaises(ValueError):verifier({'trial_id':'t1','modality':'vision','input':{}})
        self.assertFalse(backend.calls)
    def test_real_trace_minimization_preserves_wrong_value_and_never_calls_model(self):
        calls=len(self.backend.calls)
        def predicate(stage):
            claims=stage.get('resolved',{}).get('claims',[])
            wrong=any(c['field_key']=='passport.number' and c['raw_value']=='654321' for c in claims)
            return {'failed':wrong,'failure_id':'number_literal_error'}
        report=minimize_trace_failure(self.snapshot,predicate,max_calls=30,trial_path=self.root/'min.jsonl')
        self.assertEqual(report['target_failure_id'],'number_literal_error');self.assertTrue(report['task_contract_preserved'])
        self.assertNotIn('noise',{n['id'] for n in report['graph']['nodes']})
        replay=replay_selector_graph(report['graph'],self.snapshot['recorded_response'])
        self.assertTrue(predicate(replay)['failed']);self.assertEqual(report['new_model_calls'],0)
        self.assertEqual(len(self.backend.calls),calls)
    def test_finite_budget_required_for_real_research_model_calls(self):
        for timeout in [float('nan'),float('inf'),0,True]:
            with self.assertRaises(ValueError):QwenTextVerifier(Backend(),timeout=timeout)

class OracleAdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract=ProfileRegistry().compile('D01',subset=['passport.number','passport.series'])
        self.original=observation('value','123450');self.corrected=observation('value','123456')
        self.documents=[{'id':'private-doc','original_id':'source-one','split':'dev',
            'input':{'text':[self.original],'roles':{'value':{'field_key':'passport.series','party_role':'document'}}},
            'corrections':{'text':[self.corrected],'roles':{'value':{'field_key':'passport.number','party_role':'document'}}}}]
    def test_factorial_uses_real_selector_stage_and_both_factors_are_needed(self):
        backend=Backend();report=run_trace_oracle_study(self.documents,self.contract,backend,{SHA:{'passport.number':'123456'}})
        self.assertEqual(report['trial_errors'],0);self.assertEqual(report['recorded_trials'],4)
        self.assertEqual(report['complete_original_groups'],1)
        effects=report['effects'][0]
        self.assertEqual((effects['baseline'],effects['text_corrected'],effects['roles_corrected'],effects['both_corrected']),(0,0,0,1))
        self.assertEqual(effects['interaction'],1)
        for _,args,_ in backend.calls:
            prompt=json.loads(args['prompt']);self.assertIn('research_intervention',prompt)
            self.assertNotIn('private-doc',args['prompt']);self.assertNotIn('gold',prompt)
            self.assertIn(prompt['research_intervention']['text_factor'],['original','corrected'])
    def test_private_reference_does_not_enter_any_model_prompt(self):
        backend=Backend();run_trace_oracle_study(self.documents,self.contract,backend,{SHA:{'passport.number':'999999'}})
        self.assertEqual(len(backend.calls),4)
        self.assertTrue(all('999999' not in args['prompt'] for _,args,_ in backend.calls))
    def test_production_and_value_smuggling_role_maps_refused(self):
        backend=Backend();docs=copy.deepcopy(self.documents);docs[0]['split']='production'
        with self.assertRaises(ValueError):run_trace_oracle_study(docs,self.contract,backend,{SHA:{'passport.number':'123456'}})
        self.assertFalse(backend.calls)
        docs=copy.deepcopy(self.documents)
        for material in [docs[0]['input'],docs[0]['corrections']]:material['roles']['value']['raw_value']='123456'
        report=run_trace_oracle_study(docs,self.contract,backend,{SHA:{'passport.number':'123456'}})
        self.assertEqual(report['trial_errors'],4);self.assertFalse(backend.calls)

if __name__=='__main__':unittest.main()
