"""Primary-source machine-ID subset and strict evidence bindings; no NN."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness_ocr.grounding import build_candidates
from harness_ocr.profiles import ProfileRegistry
from harness_ocr.machine_ids import validate_machine_identifiers,derive_machine_context,REGISTRY_PATH,REGISTRY_SHA256


PIN='1T0331GJCGF300253'  # Constructed case at the published manual's lower boundary.


def case(pin=PIN,serial=None,manufacturer='John Deere',model='331G',prefix='',oid='o1'):
    text=prefix+'Машина; Изготовитель: '+manufacturer+'; Модель: '+model+'; PIN: '+pin
    if serial is not None:text+='; Заводской номер: '+serial
    obs={'id':oid,'text':text,'source_sha256':'a'*64,'active':True,'completed':True,
         'evidence_kind':'image_region','page':1,'page_id':'p1','source_size':[1200,200],'bbox':[0,0,1200,200]}
    contract=ProfileRegistry().compile('D05',subset=['machine.main_identifier'])
    candidate=next(iter(build_candidates([obs],contract).values()))
    return candidate['raw_value'],[obs]


class MachineIdTests(unittest.TestCase):
    def check(self,*args,**kwargs):
        raw,obs=case(*args,**kwargs)
        return validate_machine_identifiers(raw,obs)

    def test_registry_bytes_are_pinned(self):
        self.assertEqual(hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),REGISTRY_SHA256)

    def test_known_source_family_at_manual_boundary_passes_only_scoped_checks(self):
        result=self.check()
        self.assertEqual(result['validation_status'],'pass',result)
        self.assertTrue(result['checks'])
        self.assertTrue(all(set(c)=={'check_id','status','scope','reason_codes','source_ids','evidence_ids'} for c in result['checks']))
        self.assertTrue(any(c['scope']=='assigned_alpha_only_no_checksum_and_no_authenticity' for c in result['checks']))
        self.assertEqual(result['checks'][-1]['status'],'not_applicable')

    def test_factory_literal_is_checked_against_independently_observed_pin_suffix(self):
        raw,obs=case(serial='300253');before=deepcopy((raw,obs))
        result=validate_machine_identifiers(raw,obs)
        self.assertEqual(result['validation_status'],'pass',result)
        self.assertEqual((raw,obs),before)
        relation=next(c for c in result['checks'] if c['check_id'].endswith('factory_serial_relation'))
        self.assertEqual(relation['status'],'pass')
        self.assertEqual(relation['evidence_ids'],['o1'])

    def test_changed_assigned_letter_is_not_falsely_rejected_as_vin_checksum(self):
        changed=PIN[:8]+'D'+PIN[9:]
        self.assertEqual(self.check(changed)['validation_status'],'pass')

    def test_family_model_333g_has_its_own_explicit_context(self):
        changed=PIN[:3]+'333G'+PIN[7:]
        self.assertEqual(self.check(changed,model='333G')['validation_status'],'pass')

    def test_old_illustration_is_not_qualified_by_newer_manual_cover_range(self):
        raw,obs=case('1T0331GJCGF000001',serial='000001')
        result=validate_machine_identifiers(raw,obs)
        self.assertEqual(result['validation_status'],'not_checked')
        self.assertIn('machine_pin_before_registered_manual_range',result['validation_reasons'])
        self.assertEqual(next(i['raw_value'] for i in raw['items'] if i['kind']=='factory_number'),'000001')

    def test_unknown_manufacturer_model_or_wmc_is_not_promoted_by_pin_appearance(self):
        for kwargs in ({'manufacturer':'Caterpillar'},{'model':'334G'},{'pin':'1DW'+PIN[3:]}):
            with self.subTest(kwargs=kwargs):self.assertEqual(self.check(**kwargs)['validation_status'],'not_checked')

    def test_unknown_option_year_or_emission_stays_outside_registered_subset(self):
        for offset,char in ((7,'Z'),(9,'J'),(10,'G')):
            changed=PIN[:offset]+char+PIN[offset+1:]
            with self.subTest(offset=offset):self.assertEqual(self.check(changed)['validation_status'],'not_checked')

    def test_explicit_supported_shape_or_model_disagreement_fails_without_repair(self):
        examples=[PIN[:-1],PIN.lower(),PIN[:8]+'7'+PIN[9:],PIN[:13]+'A'+PIN[14:]]
        for value in examples:
            with self.subTest(value=value):self.assertEqual(self.check(value)['validation_status'],'fail')
        self.assertEqual(self.check(model='333G')['validation_status'],'fail')

    def test_mismatching_factory_is_not_replaced_by_pin_digits(self):
        raw,obs=case(serial='300254');before=deepcopy(raw)
        result=validate_machine_identifiers(raw,obs)
        self.assertEqual(result['validation_status'],'fail')
        self.assertIn('machine_pin_factory_number_disagreement',result['validation_reasons'])
        self.assertEqual(raw,before)

    def test_missing_pin_item_is_not_reconstructed_from_source_or_factory_literal(self):
        raw,obs=case(serial='300253');raw['items']=[i for i in raw['items'] if i['kind']=='factory_number']
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_omitted_visible_factory_item_prevents_partial_list_pass(self):
        raw,obs=case(serial='300253');raw['items']=[i for i in raw['items'] if i['kind']=='pin']
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_manufacturer_or_model_on_another_line_cannot_lend_context(self):
        raw,obs=case(prefix='Изготовитель: John Deere\n')
        start=obs[0]['text'].rfind('Изготовитель: John Deere; ')
        # Construct the actual extracted candidate after removing only the
        # current-line context; the preceding source row remains present.
        text=obs[0]['text'][:start]+obs[0]['text'][start+len('Изготовитель: John Deere; '):]
        obs[0]['text']=text
        contract=ProfileRegistry().compile('D05',subset=['machine.main_identifier'])
        raw=next(iter(build_candidates(obs,contract).values()))['raw_value']
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_factory_from_another_source_or_row_cannot_be_paired(self):
        raw,obs=case(serial='300253');other,other_obs=case(serial='300253',oid='o2')
        raw['items']=[next(i for i in raw['items'] if i['kind']=='pin'),next(i for i in other['items'] if i['kind']=='factory_number')]
        self.assertEqual(validate_machine_identifiers(raw,obs+other_obs)['validation_status'],'not_checked')

    def test_inactive_incomplete_duplicate_or_foreign_evidence_never_passes(self):
        for mutation in (lambda o:o.update(active=False),lambda o:o.update(completed=False),lambda o:o.update(source_sha256='not-a-sha'),
                         lambda o:o.update(evidence_kind='structured_xml'),lambda o:o.update(evidence_kind='native_pdf_text',text_layer_checks=False)):
            raw,obs=case();mutation(obs[0])
            self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')
        raw,obs=case()
        self.assertEqual(validate_machine_identifiers(raw,obs+deepcopy(obs))['validation_status'],'not_checked')

    def test_literal_label_role_and_span_tampering_is_rejected(self):
        for mutate in (lambda i:i.update(raw_value=PIN[:-1]+'4'),lambda i:i.update(party_role='engine'),
                       lambda i:i['spans'][0].update(start=0),lambda i:i['kind_spans'][0].update(start=0),
                       lambda i:i['role_spans'][0].update(end=1),lambda i:i['association'].update(line_start=1)):
            raw,obs=case();mutate(raw['items'][0])
            self.assertNotEqual(validate_machine_identifiers(raw,obs)['validation_status'],'pass')

    def test_unknown_edition_and_unsupported_layout_do_not_pass(self):
        raw,obs=case();raw['items'][0]['version']='unresolved'
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')
        raw,obs=case();obs[0]['text']+='; Другой номер: 123456'
        raw['items'][0]['association']['line_end']=len(obs[0]['text'])
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_multiple_machine_roles_or_duplicate_context_cannot_be_sliced_away(self):
        raw,obs=case();obs[0]['text']+='; Машина'
        raw['items'][0]['association']['line_end']=len(obs[0]['text'])
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')
        raw,obs=case();obs[0]['text']+='; Модель: 333G'
        raw['items'][0]['association']['line_end']=len(obs[0]['text'])
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_registry_tampering_abstains_even_with_plausible_literals(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sources.json';path.write_bytes(REGISTRY_PATH.read_bytes()+b' ')
            with patch('harness_ocr.machine_ids.REGISTRY_PATH',path):
                self.assertEqual(self.check()['validation_status'],'not_checked')

    def test_malformed_input_is_fail_closed_not_an_exception(self):
        for raw in (None,[],{},'PIN:'+PIN,{'items':[None]}):
            self.assertEqual(validate_machine_identifiers(raw,[])['validation_status'],'not_checked')
        for value in ([],{},None):
            raw,obs=case();raw['items'][0]['kind']=value
            self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')


class MachineContextIdentityTests(unittest.TestCase):
    def test_unknown_brand_and_model_are_literal_identity_not_format_qualification(self):
        raw,obs=case(manufacturer='Unregistered Manufacturer',model='UNKNOWN-1')
        self.assertEqual(derive_machine_context(raw,obs),{'manufacturer':'unregistered manufacturer','model':'UNKNOWN-1'})
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_same_literal_has_distinct_explicit_manufacturer_contexts(self):
        first,one=case();second,two=case(manufacturer='Other Manufacturer',oid='o2')
        self.assertNotEqual(derive_machine_context(first,one),derive_machine_context(second,two))

    def test_incoming_normalized_observation_has_context_without_acceptance_flag(self):
        raw,obs=case();obs[0].pop('active')
        self.assertEqual(derive_machine_context(raw,obs),{'manufacturer':'john deere','model':'331G'})
        self.assertEqual(validate_machine_identifiers(raw,obs)['validation_status'],'not_checked')

    def test_missing_manufacturer_does_not_borrow_another_observation(self):
        raw,obs=case()
        text=obs[0]['text'].replace('Изготовитель: John Deere; ','')
        obs[0]['text']=text
        contract=ProfileRegistry().compile('D05',subset=['machine.main_identifier'])
        raw=next(iter(build_candidates(obs,contract).values()))['raw_value']
        extra,extra_obs=case(oid='other')
        self.assertIsNone(derive_machine_context(raw,obs+extra_obs))

    def test_inactive_incomplete_mixed_or_forged_source_has_no_context(self):
        for mutate in (lambda o:o.update(active=False),lambda o:o.update(completed=False),
                       lambda o:o.update(source_sha256='bad')):
            raw,obs=case();mutate(obs[0]);self.assertIsNone(derive_machine_context(raw,obs))
        raw,obs=case(serial='300253');other,extra=case(serial='300253',oid='o2')
        raw['items']=[next(i for i in raw['items'] if i['kind']=='pin'),next(i for i in other['items'] if i['kind']=='factory_number')]
        self.assertIsNone(derive_machine_context(raw,obs+extra))
        raw,obs=case();raw['items'][0]['spans'][0]['start']=0
        self.assertIsNone(derive_machine_context(raw,obs))


if __name__=='__main__':unittest.main()
