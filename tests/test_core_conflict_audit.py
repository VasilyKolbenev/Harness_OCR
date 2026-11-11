"""Withdrawal of conflicting evidence retires history, never revives guesses."""
from pathlib import Path
import tempfile
import unittest
from harness_ocr.core import DocumentStore

SHA='a'*64

def obs(ident,text):
    return {'id':ident,'text':text,'source_sha256':SHA,'evidence_kind':'image_region',
            'page':1,'page_id':'p1','source_size':[800,600],'bbox':[0,0,800,600],'completed':True}

def claim(ident,value,source,field='passport.number',depends=()):
    return {'id':ident,'field_key':field,'raw_value':value,'evidence_ids':[source],
            'party_role':'document','origin':'rule','depends_on':list(depends)}

class ConflictWithdrawalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=DocumentStore(Path(self.temp.name)/'state.sqlite')
        self.store.create_document('doc',SHA,'fixture-profile')
    def tearDown(self):self.temp.cleanup()
    def seed_conflict(self):
        self.store.revise('doc',observations=[obs('sourceA','123456'),obs('sourceB','654321'),
                  obs('derivedSource','10'),obs('independentSource','safe')],
            claims=[claim('A','123456','sourceA'),claim('independent','safe','independentSource',field='independent')])
        self.store.revise('doc',claims=[claim('derived','10','derivedSource',field='derived',depends=['A'])])
        return self.store.revise('doc',claims=[claim('B','654321','sourceB')])

    def test_bad_conflict_source_and_historic_opponent_retire_without_automatic_reacceptance(self):
        conflict=self.seed_conflict()
        self.assertEqual(conflict['claims']['A']['status'],'conflict')
        state=self.store.revise('doc',invalidate=['sourceB'])
        self.assertEqual(state['claims']['B']['status'],'retracted')
        self.assertEqual(state['claims']['A']['status'],'retracted')
        self.assertEqual(state['claims']['A']['status_reason'],'resolved_conflict_requires_new_claim_or_proof')
        self.assertEqual(state['claims']['derived']['status'],'retracted')
        self.assertEqual(state['fields']['passport.number']['status'],'withheld')
        self.assertIsNone(state['fields']['passport.number']['raw_value'])
        self.assertEqual(state['fields']['independent']['raw_value'],'safe')
        self.assertTrue(state['needs_review']);self.assertFalse(state['automatic_acceptance_authorized'])
        # A repeated immutable claim ID cannot resurrect the historic decision.
        repeated=self.store.revise('doc',claims=[claim('A','123456','sourceA')])
        self.assertEqual(repeated['claims']['A']['status'],'retracted')
        # A fresh reading and fresh literal claim are eligible for support, not
        # auto acceptance. Retracted conflict B must not remain a sticky veto.
        fresh=self.store.revise('doc',observations=[obs('sourceA2','123456')],
               claims=[claim('A2','123456','sourceA2')])
        self.assertEqual(fresh['fields']['passport.number']['status'],'supported')
        self.assertFalse(fresh['fields']['passport.number']['needs_review'])
        self.assertFalse(fresh['conflicts'])
        self.assertTrue(fresh['needs_review']);self.assertFalse(fresh['automatic_acceptance_authorized'])

    def test_withdrawal_keeps_real_opposition_if_another_conflicting_read_survives(self):
        self.seed_conflict()
        self.store.revise('doc',observations=[obs('sourceB2','654321')],
             claims=[claim('B2','654321','sourceB2')])
        state=self.store.revise('doc',invalidate=['sourceB'])
        self.assertEqual(state['claims']['B']['status'],'retracted')
        self.assertEqual(state['claims']['A']['status'],'conflict')
        self.assertTrue(state['conflicts']);self.assertTrue(state['fields']['passport.number']['needs_review'])

    def test_operator_value_and_evidence_conflict_require_explicit_authorized_confirmation(self):
        state=self.seed_conflict()
        state=self.store.human_correct('doc','passport.number','123456',['sourceA'],actor='reviewer',
           expected_revision=state['revision'],authorized=True,party_role='document')
        state=self.store.revise('doc',observations=[obs('sourceB2','654321')],
               claims=[claim('B2','654321','sourceB2')])
        self.assertTrue(state['fields']['passport.number']['needs_review'])
        state=self.store.revise('doc',invalidate=['sourceB2'])
        self.assertEqual(state['fields']['passport.number']['origin'],'human')
        self.assertEqual(state['fields']['passport.number']['raw_value'],'123456')
        self.assertTrue(state['fields']['passport.number']['needs_review'])
        self.assertEqual(state['claims']['B2']['status'],'retracted')
        confirmed=self.store.human_correct('doc','passport.number','123456',['sourceA'],actor='reviewer',
           expected_revision=state['revision'],authorized=True,party_role='document')
        self.assertFalse(confirmed['fields']['passport.number']['needs_review'])

if __name__=='__main__':unittest.main()
