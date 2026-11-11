import json
import unittest
from harness_ocr.grounding import build_candidates,resolve_response
from harness_ocr.profiles import ProfileRegistry

class CorroborationTests(unittest.TestCase):
    def setUp(self):
        self.contract=ProfileRegistry().compile('D01',subset=['passport.number'])
    def candidates(self,texts):
        return build_candidates([{'id':'o'+str(i),'text':t,'completed':True} for i,t in enumerate(texts)],self.contract)
    def resolve(self,candidates,ids):
        return resolve_response(json.dumps({'proposals':[{'field_key':'passport.number','candidate_ids':ids,
             'party_role':'document'}],'reread_requests':[]}),candidates,[],self.contract)
    def test_identical_ctc_and_xiaomi_literals_merge_sources_without_join(self):
        c=self.candidates(['Номер паспорта: 123456','Номер паспорта: 123456'])
        resolved=self.resolve(c,list(c))
        self.assertEqual(resolved['claims'][0]['raw_value'],'123456')
        self.assertEqual(resolved['claims'][0]['evidence_ids'],['o0','o1'])
        self.assertEqual(len(resolved['claims'][0]['role_spans']),2)
        self.assertEqual(self.resolve(c,list(reversed(c)))['claims'][0]['id'],resolved['claims'][0]['id'])
    def test_differing_values_abstain_instead_of_digit_join_or_majority(self):
        c=self.candidates(['Номер паспорта: 123456','Номер паспорта: 654321'])
        resolved=self.resolve(c,list(c))
        self.assertFalse(resolved['claims']);self.assertEqual(resolved['rejected'][0]['reason'],'candidate_disagreement')
    def test_one_unanchored_source_cannot_borrow_another_role(self):
        c=self.candidates(['Номер паспорта: 123456','123456'])
        resolved=self.resolve(c,list(c))
        self.assertFalse(resolved['claims']);self.assertEqual(resolved['rejected'][0]['reason'],'role_anchor_not_observed')
    def test_duplicate_and_invented_ids_still_rejected(self):
        c=self.candidates(['Номер паспорта: 123456']);cid=next(iter(c))
        for ids in [[cid,cid],[cid,'invented'],[],[{}]]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):self.resolve(c,ids)

if __name__=='__main__':unittest.main()
