"""Typed field contract regressions. Synthetic evidence; no NN or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import validate as schema_validate, ValidationError

from harness_ocr.grounding import build_candidates, resolve_response, validate_literal
from harness_ocr.profiles import ProfileRegistry, project_contract_result
from harness_ocr.typed_fields import typed_supported, value_identity
from harness_ocr.core.state import DocumentStore
from harness_ocr.native import decode_mchd
from tests.test_mchd import synthetic_xml


def observation(text, oid='o1'):
    return {'id':oid,'text':text,'completed':True,'active':True,
            'source_sha256':'a'*64,'page':1,'page_id':'p1'}


class TypedFieldTests(unittest.TestCase):
    def setUp(self):self.registry=ProfileRegistry()

    def extract(self, profile, field, text):
        obs=observation(text); contract=self.registry.compile(profile,subset=[field])
        candidates=build_candidates([obs],contract)
        return obs,contract,candidates

    def one(self,profile,field,text):
        obs,contract,candidates=self.extract(profile,field,text)
        self.assertEqual(len(candidates),1,candidates)
        candidate=next(iter(candidates.values()))
        return obs,contract,candidate

    def result(self,obs,contract,candidate):
        return validate_literal(candidate['raw_value'],contract['field_definitions'][candidate['field_key']],observations=[obs])

    def test_all_seven_declarations_bind_registered_typed_contract(self):
        count=0
        for profile in self.registry.list():
            contract=self.registry.compile(profile['profile_id'])
            for key,spec in contract['field_definitions'].items():
                if 'typed_contract_id' in spec:
                    count+=1
                    prop=contract['json_schema']['properties']['fields']['properties'][key]['properties']
                    self.assertIn('anyOf',prop['raw_value'])
                    self.assertIn('anyOf',prop['canonical_value'])
                    self.assertFalse(contract['automatic_acceptance_authorized'])
        self.assertEqual(count,7)

    def test_party_id_has_independent_kind_role_version_and_value_spans(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        raw=item['raw_value'];self.assertIsInstance(raw,dict)
        self.assertTrue(typed_supported(raw,[obs]))
        self.assertEqual(raw['items'][0]['kind'],'inn')
        self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')
        self.assertNotEqual(raw['items'][0]['spans'],raw['items'][0]['kind_spans'])
        self.assertEqual(validate_literal(raw,c['field_definitions']['principal.identifier'])['validation_status'],'not_checked')

    def test_wrong_party_and_unlabelled_digit_count_do_not_select_kind(self):
        for text in ['Представитель ИНН: 7707083893','Доверитель документ: 7707083893',
                     'Доверитель ИНН представителя: 7707083893']:
            self.assertEqual(self.extract('D04','principal.identifier',text)[2],{})

    def test_postfixed_role_label_is_supported_without_digit_inference(self):
        obs,c,item=self.one('D04','representative.identifier','ИНН представителя: 7707083893')
        self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')

    def test_conflicting_identifiers_in_separate_rows_remain_review(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893\nДоверитель ИНН: 7736050003')
        result=self.result(obs,c,item)
        self.assertEqual(result['validation_status'],'not_checked')
        self.assertIn('multiple_identifier_values_for_one_kind',' '.join(result['validation_reasons']))

    def test_inn_checksum_failure_is_not_repaired(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083894')
        result=self.result(obs,c,item)
        self.assertEqual(result['validation_status'],'fail')
        self.assertEqual(result['canonical_value']['items'][0]['raw_value'],'7707083894')

    def test_rf_passport_leading_zeroes_are_lossless(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель паспорт РФ: 00 01 000023')
        result=self.result(obs,c,item)
        self.assertEqual(result['validation_status'],'pass')
        self.assertEqual(result['canonical_value']['items'][0]['raw_value'],'00 01 000023')
        self.assertEqual(result['transform_ids'],[])

    def test_ogrn_and_ogrnip_explicit_labels_use_distinct_checksums(self):
        for kind,value in [('ОГРН','1027700132195'),('ОГРНИП','304500116000157')]:
            obs,c,item=self.one('D04','principal.identifier','Доверитель '+kind+': '+value)
            self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')
            obs,c,item=self.one('D04','principal.identifier','Доверитель '+kind+': '+value[:-1]+'0')
            self.assertEqual(self.result(obs,c,item)['validation_status'],'fail')

    def test_kind_role_and_span_tampering_rejected_by_reconstruction(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        for mutate in [lambda x:x['items'][0].update(kind='ogrn',version='ogrn-label-v1'),
                       lambda x:x['items'][0].update(party_role='representative'),
                       lambda x:x['items'][0]['spans'][0].update(start=15),
                       lambda x:x['items'][0].update(role_spans=x['items'][0]['spans'])]:
            raw=deepcopy(item['raw_value']);mutate(raw)
            self.assertFalse(typed_supported(raw,[obs]))

    def test_unknown_version_and_json_stuffed_string_remain_review(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        raw=deepcopy(item['raw_value']);raw['contract_version']='999'
        self.assertEqual(validate_literal(raw,c['field_definitions']['principal.identifier'],observations=[obs])['validation_status'],'not_checked')
        self.assertEqual(validate_literal(json.dumps(item['raw_value']),c['field_definitions']['principal.identifier'],observations=[obs])['validation_status'],'not_checked')

    def test_machine_family_requires_explicit_kind_and_supported_version(self):
        for profile in ['D05','D06']:
            self.assertEqual(self.extract(profile,'machine.main_identifier','Машина номер: 12345678901234567')[2],{})
            obs,c,item=self.one(profile,'machine.main_identifier','Машина VIN: 12345678901234567')
            self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')
            obs,c,item=self.one(profile,'machine.main_identifier','ГОСТ 33990-2016\nМашина VIN: XTA210990Y2765432')
            self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')

    def test_vin_rejects_invalid_letters_and_wrong_length(self):
        for value in ['XTA210990Y276543I','12345678']:
            obs,c,item=self.one('D05','machine.main_identifier','ГОСТ 33990-2016\nМашина VIN: '+value)
            self.assertEqual(self.result(obs,c,item)['validation_status'],'fail')

    def test_machine_engine_number_does_not_become_main_identifier(self):
        self.assertEqual(self.extract('D05','machine.main_identifier','Машина заводской номер двигателя: 00013')[2],{})

    def test_pin_and_factory_are_preserved_without_inventing_manufacturer_format(self):
        for label,kind in [('PIN','pin'),('заводской номер','factory_number')]:
            obs,c,item=self.one('D05','machine.main_identifier','Машина '+label+': 000013-А' if kind=='factory_number' else 'Машина PIN: 000013-A')
            self.assertEqual(item['raw_value']['items'][0]['kind'],kind)
            self.assertEqual(self.result(obs,c,item)['validation_status'],'not_checked')

    @staticmethod
    def table(rows):return 'Сведения о собственниках\nСобственник | Дата регистрации | Основание\n'+'\n'.join(rows)

    def test_ownership_rows_columns_and_repeated_legitimate_rows_survive(self):
        row='ООО Пример | 01.02.2024 | Договор 001'
        obs,c,item=self.one('D05','ownership.entries',self.table([row,row]))
        raw=item['raw_value'];self.assertEqual(len(raw['items']),2)
        self.assertEqual([x['raw_value'] for x in raw['items']],[row,row])
        self.assertNotEqual(raw['items'][0]['spans'],raw['items'][1]['spans'])
        self.assertEqual([x['column'] for x in raw['items'][0]['cells']],['owner','date','document'])
        self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')
        omitted=deepcopy(raw);omitted['items'].pop()
        self.assertFalse(typed_supported(omitted,[obs]))

    def test_ownership_crossrow_and_reordering_abstain(self):
        obs,c,item=self.one('D05','ownership.entries',self.table(['Альфа | 01.02.2024 | Договор 001','Бета | 03.04.2025 | Договор 002']))
        raw=deepcopy(item['raw_value']);raw['items'][0]['cells'][1]=deepcopy(raw['items'][1]['cells'][1])
        self.assertFalse(typed_supported(raw,[obs]))
        raw=deepcopy(item['raw_value']);raw['items'].reverse()
        self.assertFalse(typed_supported(raw,[obs]))

    def test_ownership_ambiguous_or_incomplete_layout_has_no_partial_prefix(self):
        for text in [self.table(['Альфа | 01.02.2024 | Договор 001','Бета | 03.04.2025']),
                     'Сведения о собственниках\nСобственник Дата регистрации\nАльфа 01.02.2024',
                     self.table(['Альфа | 01.02.2024 | Договор 001','Бета неполная строка'])]:
            self.assertEqual(self.extract('D05','ownership.entries',text)[2],{})

    def test_ownership_invalid_calendar_does_not_pass(self):
        obs,c,item=self.one('D05','ownership.entries',self.table(['Альфа | 31.02.2024 | Договор 001']))
        self.assertEqual(self.result(obs,c,item)['validation_status'],'fail')

    def test_bank_types_are_labelled_and_treasury_leading_zeroes_not_bank_checksum(self):
        text='ЕКС\nПолучатель казначейский счет: 00000000000000000001\nПолучатель банковский счет ЕКС: 40102810000000010001'
        obs,c,item=self.one('D07','payee.accounts',text)
        rows=item['raw_value']['items'];self.assertEqual([x['kind'] for x in rows],['treasury','eks'])
        self.assertEqual(rows[0]['raw_value'],'00000000000000000001')
        self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')

    def test_bank_payer_wronglabel_or_unknownversion_not_borrowed(self):
        for text in ['ЕКС\nПлательщик казначейский счет: 00000000000000000001',
                     'Получатель счет: 40702810900000002851']:
            self.assertEqual(self.extract('D07','payee.accounts',text)[2],{})

    def test_bank_checksum_only_under_explicit_contract_and_same_row_bik(self):
        # Generated arithmetic fixture: checksum digit is computed here only to
        # make a valid input; the extraction path never generates/repairs digits.
        bik='044525225';base='40702810000000000001'
        values=[base[:8]+str(n)+base[9:] for n in range(10)]
        good=next(v for v in values if sum(int(n)*[7,1,3][i%3] for i,n in enumerate(bik[-3:]+v))%10==0)
        obs,c,item=self.one('D07','payee.accounts','Банк России 515\nПолучатель БИК: '+bik+'; расчетный счет: '+good)
        result=self.result(obs,c,item)
        self.assertEqual(result['validation_status'],'not_checked')
        self.assertIn('account_opening_location_not_established',' '.join(result['validation_reasons']))
        self.assertFalse(any('mismatch' in reason for reason in result['validation_reasons']))
        bad=good[:-1]+str((int(good[-1])+1)%10)
        obs,c,item=self.one('D07','payee.accounts','Банк России 515\nПолучатель БИК: '+bik+'; расчетный счет: '+bad)
        result=self.result(obs,c,item)
        self.assertEqual(result['validation_status'],'not_checked')
        self.assertIn('account_opening_location_not_established',' '.join(result['validation_reasons']))
        obs,c,item=self.one('D07','payee.accounts','Банк России 515\nПолучатель БИК: '+bik+'\nПолучатель расчетный счет: '+good)
        self.assertEqual(self.result(obs,c,item)['validation_status'],'not_checked')

    def test_bank_subtype_required_accounts_and_unknown_account_cell_block_pass(self):
        for text,reason in [('Получатель казначейский счет: 00000000000000000001','treasury_subtype'),
                            ('Получатель банковский счет ЕКС: 40102810000000010001','treasury_subtype'),
                            ('Получатель корреспондентский счет: 30101810400000000225','bank_subtype')]:
            obs,c,item=self.one('D07','payee.accounts',text)
            result=self.result(obs,c,item)
            self.assertEqual(result['validation_status'],'not_checked')
            self.assertIn(reason,' '.join(result['validation_reasons']))
        obs,c,item=self.one('D07','payee.accounts','Получатель казначейский счет: 00000000000000000001\nПолучатель банковский счет ЕКС: 40102810000000010001\nПолучатель счет: 00000000000000000002')
        self.assertEqual(self.result(obs,c,item)['validation_status'],'not_checked')

    def test_oms_version_and_kind_are_observed_not_inferred_from_length(self):
        for text in ['Застрахованное лицо номер полиса: 0000000000000001',
                     'Полис ОМС единого образца\nЗастрахованное лицо номер бланка: 0000000000000001']:
            self.assertEqual(self.extract('D10','policy.number',text)[2],{})
        obs,c,item=self.one('D10','policy.number','Полис ОМС единого образца\nЗастрахованное лицо ЕНП: 0000000000000001')
        self.assertEqual(item['raw_value']['items'][0]['raw_value'],'0000000000000001')
        self.assertTrue(typed_supported(item['raw_value'],[obs]))
        self.assertEqual(self.result(obs,c,item)['validation_status'],'fail')
        obs,c,item=self.one('D10','policy.number','Застрахованное лицо ЕНП: 0000000000000018')
        self.assertEqual(self.result(obs,c,item)['validation_status'],'pass')
        for marker in ['Полис старого образца','Версия 999']:
            obs,c,item=self.one('D10','policy.number',marker+'\nЗастрахованное лицо ЕНП: 0000000000000018')
            self.assertEqual(self.result(obs,c,item)['validation_status'],'not_checked')

    def test_retraction_and_incomplete_source_invalidate_all_typed_support(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        inactive={**obs,'active':False};incomplete={**obs,'completed':False}
        self.assertFalse(typed_supported(item['raw_value'],[inactive]))
        self.assertFalse(typed_supported(item['raw_value'],[incomplete]))

    def test_durable_core_support_identity_and_retraction_use_all_typed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            store=DocumentStore(Path(temp)/'state.sqlite')
            store.create_document('typed-doc','a'*64,'D04',{'field_keys':['principal.identifier']})
            obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
            claims=[];observations=[]
            for oid in ['o1','o2']:
                source={**observation(obs['text'],oid),'evidence_kind':'image_region','bbox':[0,0,100,100],'source_size':[100,100]}
                source.pop('active');observations.append(source)
                candidate=next(iter(build_candidates([source],c).values()))
                selection=resolve_response(json.dumps({'proposals':[{'field_key':'principal.identifier','candidate_ids':[candidate['id']],'party_role':'principal'}],'reread_requests':[]}),{candidate['id']:candidate},[],c)
                claims.extend(selection['claims'])
            state=store.revise('typed-doc',observations=observations,claims=claims)
            self.assertEqual(state['fields']['principal.identifier']['status'],'supported')
            self.assertEqual(state['conflicts'],[])
            self.assertIsInstance(state['fields']['principal.identifier']['raw_value'],dict)
            state=store.revise('typed-doc',invalidate=['o1','o2'])
            self.assertIsNone(state['fields']['principal.identifier']['raw_value'])
            self.assertTrue(all(x['status']=='retracted' for x in state['claims'].values()))

    def test_structured_selector_and_public_schema_preserve_nested_data(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        selected=resolve_response(json.dumps({'proposals':[{'field_key':item['field_key'],'candidate_ids':[item['id']], 'party_role':'principal'}],'reread_requests':[]}),{item['id']:item},[],c)
        raw=selected['claims'][0]['raw_value'];self.assertIsInstance(raw,dict)
        runtime={'needs_review':True,'fields':{'principal.identifier':{'raw_value':raw,'canonical_value':deepcopy(raw),'party_role':'principal',
                 'extraction_status':'observed','validation_status':'pass','decision':'review','evidence_ids':['o1']}}}
        projected=project_contract_result(c,runtime);schema_validate(projected,c['json_schema'])
        self.assertIsInstance(projected['fields']['principal.identifier']['raw_value'],dict)
        projected['fields']['principal.identifier']['raw_value']['items'][0]['version']='guessed'
        with self.assertRaises(ValidationError):schema_validate(projected,c['json_schema'])

    def test_identity_ignores_source_location_but_preserves_duplicate_ordered_rows(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        other=observation(obs['text'],'o2');other_item=next(iter(build_candidates([other],c).values()))
        self.assertEqual(value_identity(item['raw_value']),value_identity(other_item['raw_value']))
        obs,c,item=self.one('D05','ownership.entries',self.table(['Альфа | 01.02.2024 | Договор 001']*2))
        raw=deepcopy(item['raw_value']);raw['items'].pop()
        self.assertNotEqual(value_identity(item['raw_value']),value_identity(raw))

    def test_unregistered_native_association_is_never_a_text_label_bypass(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        raw=deepcopy(item['raw_value']);raw['items'][0]['association']={'kind':'validated_native_xml','native_validation':{}}
        self.assertFalse(typed_supported(raw,[obs]))

    def test_malformed_nested_types_return_review_without_validator_exception(self):
        obs,c,item=self.one('D04','principal.identifier','Доверитель ИНН: 7707083893')
        spec=c['field_definitions']['principal.identifier']
        for mutate in [lambda x:x.update(contract_id=[]),lambda x:x['items'][0].update(kind={}),
                       lambda x:x['items'][0].update(party_role=[]),lambda x:x['items'][0].update(version=[])]:
            raw=deepcopy(item['raw_value']);mutate(raw)
            self.assertFalse(typed_supported(raw,[obs]))
            self.assertEqual(validate_literal(raw,spec,observations=[obs])['validation_status'],'not_checked')

    def test_declared_mchd_iso_date_transform_keeps_raw_and_requires_exact_mapping(self):
        decoded=decode_mchd(synthetic_xml())
        contract=self.registry.compile('D04')
        for field,raw,canonical in [('power.issue_date','2026-01-02','02.01.2026'),('power.valid_until','2027-03-04','04.03.2027')]:
            spec=contract['field_definitions'][field]
            result=validate_literal(raw,spec,observations=decoded['observations'])
            self.assertEqual(result['validation_status'],'pass')
            self.assertEqual(result['canonical_value'],canonical)
            self.assertEqual(result['transform_ids'],['mchd-iso-date-v1'])
            source=next(o for o in decoded['observations'] if o['native_proof']['mapping_id']==field)
            self.assertEqual(source['text'],raw)
            for changes in [{'active':False},{'completed':False},{'evidence_kind':'image_region'},{'schema_version':'UNKNOWN'},
                            {'native_proof':{**source['native_proof'],'mapping_id':'power.other_date'}},
                            {'native_proof':{**source['native_proof'],'xpath':'/another/@date'}},
                            {'native_validation':{**source['native_validation'],'mapper_sha256':'0'*64}}]:
                changed={**deepcopy(source),**changes}
                self.assertNotEqual(validate_literal(raw,spec,observations=[changed])['validation_status'],'pass')
            undeclared={**spec,'normalization_transform_ids':[]}
            self.assertNotEqual(validate_literal(raw,undeclared,observations=[source])['validation_status'],'pass')
            self.assertNotEqual(validate_literal(raw,spec,observations=[observation(raw)])['validation_status'],'pass')
        other=contract['field_definitions']['power.issue_date']|{'key':'passport.issue_date'}
        self.assertNotEqual(validate_literal('2026-01-02',other,observations=decoded['observations'])['validation_status'],'pass')
        plain=validate_literal('02.01.2026',contract['field_definitions']['power.issue_date'])
        self.assertEqual(plain['validation_status'],'pass');self.assertEqual(plain['transform_ids'],[])


if __name__=='__main__':unittest.main()
