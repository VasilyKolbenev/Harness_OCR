"""Synthetic, known-gold EMCHD_1 cases validated by the original official XSD.

No real person's XML, signature, key, portal export or runtime data is included.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lxml import etree as ET

from harness_ocr.native import (decode_mchd, NativeMchdError,
    verify_native_result, verify_native_observation, verify_typed_xml_item,
    verify_typed_xml_value)
from harness_ocr.native import mchd


def synthetic_xml(*, principals=1, representatives=1, organization=False, machine=False):
    """Invented names and explicit test IDs, not an exported real document."""
    ns = mchd.NAMESPACE
    def tag(name): return '{' + ns + '}' + name
    root = ET.Element(tag('Доверенность'), nsmap={None: ns}, ВерсФорм='EMCHD_1',
                      ПрЭлФорм='00000000', ИдФайл='SYNTHETIC_TEST_ONLY')
    power = ET.SubElement(ET.SubElement(root, tag('Документ')), tag('Довер'))
    info = ET.SubElement(power, tag('СвДов'), ВидДовер='1', ПрПередов='1', ВнНомДовер='  test № 1  ',
        НомДовер='00000000-0000-4000-8000-000000000001', ДатаВыдДовер='2026-01-02', СрокДейст='2027-03-04')
    ET.SubElement(info, tag('СведСист')).text = 'https://example.invalid/synthetic-status'
    for i in range(principals):
        wrapper = ET.SubElement(power, tag('СвДоверит'), ТипДоверит='1' if organization else '4')
        body = ET.SubElement(wrapper, tag('Доверит'))
        if organization:
            entity = ET.SubElement(body, tag('РосОргДовер'), ЕИОУК='0', ЕИОФЛ='1', ЕИОИП='0')
            ET.SubElement(entity, tag('СвРосОрг'), НаимОрг='СИНТЕТИЧЕСКИЙ ДОВЕРИТЕЛЬ & ТЕСТ',
                          ИННЮЛ='7707083893', ОГРН='1027700132195', КПП='770701001')
            leader = ET.SubElement(entity, tag('ЛицоБезДов'), ПолнЮЛ='1')
            leader_fl = ET.SubElement(leader, tag('СвФЛ'), ИННФЛ='500100732259', СНИЛС='112-233-445 95')
            leader_name = ET.SubElement(leader_fl, tag('СведФЛ'))
            ET.SubElement(leader_name, tag('ФИО'), Фамилия='НЕПРЕДСТАВИТЕЛЬ', Имя='РУКОВОДИТЕЛЬ')
        else:
            entity = ET.SubElement(body, tag('ФЛДовер'), ИННФЛ='500100732259', СНИЛС='112-233-445 95')
            details = ET.SubElement(entity, tag('СведФЛ'))
            ET.SubElement(details, tag('ФИО'), Фамилия='ТЕСТОВЫЙ'+str(i+1), Имя='ДОВЕРИТЕЛЬ', Отчество='СИНТЕТИЧЕСКИЙ')
    for i in range(representatives):
        wrapper = ET.SubElement(power, tag('СвУпПред'), ТипПред='3')
        entity = ET.SubElement(ET.SubElement(wrapper, tag('Пред')), tag('СведФизЛ'),
                               ИННФЛ='500100732259', СНИЛС='112-233-445 95')
        details = ET.SubElement(entity, tag('СведФЛ'), ДатаРожд='1990-01-01')
        ET.SubElement(details, tag('ФИО'), Фамилия='ТЕСТОВЫЙ'+str(i+1), Имя='ПРЕДСТАВИТЕЛЬ')
        ET.SubElement(details, tag('УдЛичнФЛ'), КодВидДок='21', СерНомДок='45 01 000001',
                      ДатаДок='2020-01-02', ВыдДок='СИНТЕТИЧЕСКИЙ ОРГАН')
    authority = ET.SubElement(power, tag('СвПолн'), ТипПолн='1' if machine else '0', ПрСовмПолн='1')
    if machine:
        coded = ET.SubElement(authority, tag('МашПолн'), КодПолн='TEST_001', НаимПолн='ТЕСТОВОЕ ПОЛНОМОЧИЕ')
        limit = ET.SubElement(coded, tag('ОгрСвПолн'), ИдОгр='1', КодОгр='TEST_LIMIT', НаимОгр='ОГРАНИЧЕНИЕ')
        ET.SubElement(limit, tag('ТексЗначОгр')).text = 'без права получения денежных средств'
    else:
        ET.SubElement(authority, tag('ТекстПолн')).text = '  Подавать заявления; НЕ получать деньги.\nС ограничениями.  '
    return ET.tostring(root, encoding='UTF-8', xml_declaration=True)


class MchdAdapterTests(unittest.TestCase):
    def test_original_official_xsd_pin_and_manifest(self):
        path = mchd.DATA_DIR / mchd.XSD_NAME
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), mchd.SCHEMA_SHA256)
        manifest = json.loads((mchd.DATA_DIR/'mchd-schema-registry.json').read_text(encoding='utf8'))
        self.assertEqual(manifest['assets'][0]['sha256'], mchd.SCHEMA_SHA256)
        self.assertEqual(manifest['assets'][0]['dependencies'], [])
        self.assertEqual(manifest['assets'][0]['modified'], False)

    def test_valid_attributes_text_and_original_source_proof(self):
        source = synthetic_xml()
        result = decode_mchd(source)
        self.assertEqual(result['native_bridge'], mchd.BRIDGE)
        self.assertEqual(result['source_sha256'], hashlib.sha256(source).hexdigest())
        self.assertEqual(result['structural_validation_status'], 'pass')
        self.assertFalse(result['signature_verified'])
        self.assertTrue(all(v == 'not_verified' for v in result['verification'].values()))
        claims = {c['field_key']: c for c in result['native_claims']}
        self.assertEqual(claims['power.issue_date']['raw_value'], '2026-01-02')
        self.assertEqual(claims['power.authority_literal']['raw_value'],
                         '  Подавать заявления; НЕ получать деньги.\nС ограничениями.  ')
        self.assertEqual(result['native_model']['internal_number'], '  test № 1  ')
        self.assertEqual(claims['principal.identity']['raw_value'], 'ТЕСТОВЫЙ1 ДОВЕРИТЕЛЬ СИНТЕТИЧЕСКИЙ')
        self.assertEqual(claims['representative.identity']['raw_value'], 'ТЕСТОВЫЙ1 ПРЕДСТАВИТЕЛЬ')
        self.assertTrue(verify_native_result(result, source))
        for obs in result['observations']:
            self.assertEqual(obs['namespace_map'], {'m': mchd.NAMESPACE})
            self.assertTrue(verify_native_observation({**obs, 'active': True, 'created_revision': 1}, source))
            self.assertNotIn('page', obs)
            self.assertNotIn('bbox', obs)
            self.assertTrue(obs['xpath'].startswith('/m:Доверенность[1]/m:Документ[1]/m:Довер[1]/'))

    def test_typed_attribute_ids_do_not_use_fabricated_labels(self):
        result = decode_mchd(synthetic_xml(organization=True))
        fields = {c['field_key']: c for c in result['native_claims']}
        self.assertEqual(fields['principal.identity']['raw_value'], 'СИНТЕТИЧЕСКИЙ ДОВЕРИТЕЛЬ & ТЕСТ')
        self.assertNotIn('НЕПРЕДСТАВИТЕЛЬ', fields['representative.identity']['raw_value'])
        principal = fields['principal.identifier']['raw_value']['items']
        self.assertEqual([i['kind'] for i in principal], ['inn', 'ogrn', 'kpp'])
        for claim in (fields['principal.identifier'], fields['representative.identifier']):
            for item in claim['raw_value']['items']:
                self.assertTrue(verify_typed_xml_item(item, result['observations']))
                changed = deepcopy(item); changed['party_role'] = 'notary'
                self.assertFalse(verify_typed_xml_item(changed, result['observations']))
        self.assertFalse(any('ИНН' in obs['text'] for obs in result['observations']))

    def test_all_official_party_type_branches_are_explicit_and_tested(self):
        ns = mchd.NS
        def tag(name):return '{'+mchd.NAMESPACE+'}'+name
        def name(parent):
            ET.SubElement(ET.SubElement(parent, tag('СведФЛ')), tag('ФИО'), Фамилия='ТЕСТ', Имя='ПЕРСОНА')
        for code, branch_name in mchd.PRINCIPAL_BRANCHES.items():
            with self.subTest(principal_type=code):
                root=ET.fromstring(synthetic_xml())
                wrapper=root.find('m:Документ/m:Довер/m:СвДоверит',ns)
                wrapper.set('ТипДоверит',code);container=wrapper.find('m:Доверит',ns);container.clear()
                if code=='1':
                    branch=ET.SubElement(container,tag(branch_name),ЕИОУК='0',ЕИОФЛ='1',ЕИОИП='0')
                    ET.SubElement(branch,tag('СвРосОрг'),НаимОрг='ТЕСТ РОССИЙСКОЙ ОРГАНИЗАЦИИ',КПП='770701001')
                    name(ET.SubElement(ET.SubElement(branch,tag('ЛицоБезДов'),ПолнЮЛ='1'),tag('СвФЛ')))
                elif code=='2':
                    branch=ET.SubElement(container,tag(branch_name))
                    ET.SubElement(branch,tag('СвИнОрг'),НаимИО='ТЕСТ ИНОСТРАННОЙ ОРГАНИЗАЦИИ')
                    name(ET.SubElement(branch,tag('СвРукОП')))
                elif code=='3':
                    name(ET.SubElement(container,tag(branch_name),ИННФЛ='500100732259',ОГРНИП='304500116000157',СНИЛС='000-000-000 00'))
                else:
                    name(ET.SubElement(container,tag(branch_name),СНИЛС='000-000-000 00'))
                result=decode_mchd(ET.tostring(root,encoding='UTF-8'))
                self.assertEqual(result['native_model']['principals'][0]['branch'],branch_name)
        for code, branch_name in mchd.REPRESENTATIVE_BRANCHES.items():
            with self.subTest(representative_type=code):
                root=ET.fromstring(synthetic_xml())
                wrapper=root.find('m:Документ/m:Довер/m:СвУпПред',ns)
                wrapper.set('ТипПред',code);container=wrapper.find('m:Пред',ns);container.clear()
                if code in {'1','4'}:
                    ET.SubElement(container,tag(branch_name),НаимОрг='ТЕСТ ПРЕДСТАВИТЕЛЯ ОРГАНИЗАЦИИ',КПП='770701001')
                elif code=='5':
                    ET.SubElement(container,tag(branch_name),НаимИО='ТЕСТ ПРЕДСТАВИТЕЛЯ ИНОСТРАННОЙ ОРГАНИЗАЦИИ')
                elif code=='2':
                    name(ET.SubElement(container,tag(branch_name),ИННФЛ='500100732259',ОГРНИП='304500116000157',СНИЛС='000-000-000 00'))
                else:
                    name(ET.SubElement(container,tag(branch_name),ИННФЛ='500100732259',СНИЛС='000-000-000 00'))
                result=decode_mchd(ET.tostring(root,encoding='UTF-8'))
                self.assertEqual(result['native_model']['representatives'][0]['branch'],branch_name)

    def test_typed_complete_value_rejects_omission_reorder_duplicate_crossparty(self):
        from harness_ocr.typed_fields import typed_supported
        result=decode_mchd(synthetic_xml(organization=True))
        fields={c['field_key']:c for c in result['native_claims']}
        raw=fields['principal.identifier']['raw_value']
        self.assertTrue(verify_typed_xml_value(raw,result['observations']))
        selected=[o for o in result['observations'] if o['id'] in fields['principal.identifier']['evidence_ids']]
        self.assertTrue(typed_supported(raw,selected))
        for changed in (dict(raw,items=raw['items'][:1]),dict(raw,items=list(reversed(raw['items']))),
                        dict(raw,items=raw['items']+raw['items'][:1])):
            self.assertFalse(verify_typed_xml_value(changed,result['observations']))
            self.assertFalse(typed_supported(changed,result['observations']))
        changed=deepcopy(raw);changed['items'][0]=fields['representative.identifier']['raw_value']['items'][0]
        self.assertFalse(verify_typed_xml_value(changed,result['observations']))

    def test_unknown_identity_document_is_preserved_for_review_without_partial_claim(self):
        source=synthetic_xml().replace('КодВидДок="21"'.encode(),'КодВидДок="10"'.encode())
        result=decode_mchd(source)
        party=result['native_model']['representatives'][0]
        self.assertEqual(party['unmapped_identifier_documents'][0]['attributes']['КодВидДок'],'10')
        self.assertFalse(any(c['field_key']=='representative.identifier' for c in result['native_claims']))
        self.assertTrue(any('identity_document_kind_not_registered' in r for r in result['review_reasons']))

    def test_prefixes_encodings_are_resolved_and_raw_source_hash_is_encoding_specific(self):
        source=synthetic_xml();root=ET.fromstring(source)
        rewritten=ET.Element(root.tag,nsmap={'exact':mchd.NAMESPACE},**dict(root.attrib))
        rewritten.extend(root)
        prefixed=ET.tostring(rewritten,encoding='UTF-16',xml_declaration=True)
        result=decode_mchd(prefixed)
        self.assertNotEqual(result['source_sha256'],hashlib.sha256(source).hexdigest())
        self.assertEqual(result['observations'][0]['xpath'],decode_mchd(source)['observations'][0]['xpath'])

    def test_multiple_parties_keep_all_lists_without_first_party_claim(self):
        result = decode_mchd(synthetic_xml(principals=2, representatives=2))
        self.assertEqual(len(result['native_model']['principals']), 2)
        self.assertEqual(len(result['native_model']['representatives']), 2)
        self.assertEqual(len([c for c in result['field_candidates'] if c['field_key']=='principal.identity']), 2)
        self.assertFalse(any(c['party_role'] in {'principal', 'representative'} for c in result['native_claims']))
        self.assertEqual(len(result['review_reasons']), 2)

    def test_machine_authority_keeps_restrictions_without_partial_literal(self):
        result = decode_mchd(synthetic_xml(machine=True))
        self.assertEqual(result['native_model']['machine_powers'][0]['restrictions'][0]['values'][0]['raw_value'],
                         'без права получения денежных средств')
        self.assertFalse(any(c['field_key']=='power.authority_literal' for c in result['native_claims']))
        self.assertTrue(result['review_reasons'])

    def assert_rejected(self, source, reason):
        with self.assertRaisesRegex(NativeMchdError, reason):
            decode_mchd(source)

    def test_wrong_format_version_and_unsupported_allowlist(self):
        self.assert_rejected(synthetic_xml().replace(b'EMCHD_1', b'EMCHD_2'), 'namespace_or_envelope')
        self.assert_rejected(synthetic_xml().replace('ВерсФорм="EMCHD_1"'.encode(), 'ВерсФорм="003"'.encode()), 'version_not_registered')
        with self.assertRaisesRegex(NativeMchdError, 'unregistered_allowed_version'):
            decode_mchd(synthetic_xml(), allowed_versions=('guessed-v2',))
        with self.assertRaisesRegex(NativeMchdError, 'version_not_registered'):
            decode_mchd(synthetic_xml(), allowed_versions=())

    def test_same_local_name_wrong_namespace_and_extra_attribute(self):
        source = synthetic_xml()
        self.assert_rejected(source.replace('ТекстПолн>'.encode(), 'ТекстПолн xmlns="urn:attacker">'.encode(), 1), 'xsd_validation_failed')
        self.assert_rejected(source.replace('СвПолн ТипПолн'.encode(), 'СвПолн invented="true" ТипПолн'.encode()), 'xsd_validation_failed')

    def test_xsd_valid_but_discriminator_wrong_role_is_rejected(self):
        source = synthetic_xml().replace('ТипПред="3"'.encode(), 'ТипПред="1"'.encode())
        self.assertTrue(mchd._load_schema().validate(ET.fromstring(source)))
        self.assert_rejected(source, 'party_type_branch_mismatch')
        source = synthetic_xml().replace('ТипДоверит="4"'.encode(), 'ТипДоверит="1"'.encode())
        self.assertTrue(mchd._load_schema().validate(ET.fromstring(source)))
        self.assert_rejected(source, 'party_type_branch_mismatch')

    def test_unsafe_dtd_entities_dynamic_schema_xinclude_signature_envelope(self):
        source = synthetic_xml()
        payload = b'<!DOCTYPE a [<!ENTITY secret SYSTEM "file:///C:/Windows/win.ini">]>'
        self.assert_rejected(source.replace(b'?>', b'?>'+payload, 1), 'dtd_forbidden')
        payload = b'<!DOCTYPE a [<!ENTITY e "entity value">]>'
        attack = source.replace(b'?>', b'?>'+payload, 1).replace('Подавать'.encode(), b'&e;')
        self.assert_rejected(attack, 'dtd_forbidden')
        attack = source.replace('ВерсФорм='.encode(), b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="urn:x https://attacker.invalid/schema.xsd" '+ 'ВерсФорм='.encode(), 1)
        self.assert_rejected(attack, 'dynamic_schema_or_type_forbidden')
        for inserted in (b'<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"/>',
                         b'<xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="file:///secret"/>'):
            self.assert_rejected(source.replace('</Довер>'.encode(), inserted+'</Довер>'.encode()), 'signature_or_xinclude_not_supported')
        wrapped = b'<Envelope>'+source.split(b'?>',1)[1]+b'</Envelope>'
        self.assert_rejected(wrapped, 'root_namespace_or_envelope_not_supported')
        self.assert_rejected(b'PK\x03\x04signed container', 'xml_invalid_or_unsafe')

    def test_notarial_flags_delegated_unsupported_variants_fail_closed(self):
        self.assert_rejected(synthetic_xml().replace('ПрЭлФорм="00000000"'.encode(), 'ПрЭлФорм="00100000"'.encode()), 'element_flags_variant_not_registered')
        self.assert_rejected(synthetic_xml().replace('ВидДовер="1"'.encode(), 'ВидДовер="2"'.encode()), 'power_kind_variant_not_registered')

    def test_tampered_pinned_schema_fails_even_after_valid_prior_validation(self):
        self.assertEqual(decode_mchd(synthetic_xml())['structural_validation_status'], 'pass')
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, mchd.XSD_NAME).write_bytes((mchd.DATA_DIR/mchd.XSD_NAME).read_bytes()+b'\n')
            with patch.object(mchd, 'DATA_DIR', Path(tmp)):
                self.assert_rejected(synthetic_xml(), 'schema_integrity_failure')

    def test_replay_rejects_forged_values_roles_paths_schema_proof(self):
        source = synthetic_xml(); result = decode_mchd(source)
        for key, value in [('text','changed'), ('xpath','/m:Доверенность[1]/@ИННФЛ'),
                           ('party_role','notary'), ('schema_sha256','0'*64)]:
            obs = deepcopy(result['observations'][0]); obs[key]=value
            self.assertFalse(verify_native_observation(obs, source))
        result['native_claims'][0]['raw_value'] = 'forged'
        self.assertFalse(verify_native_result(result, source))

    def test_ingest_requires_explicit_D04_profile_and_generic_stays_unqualified(self):
        from harness_ocr.ingest import decode
        from harness_ocr.settings import Settings
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'power.xml'; source.write_bytes(synthetic_xml())
            settings = Settings(root/'state', root/'models', device='cpu', dtype='float32')
            self.assertEqual(decode(source, root/'rasters', settings)['native_bridge'], 'generic_xml_unqualified')
            native = decode(source, root/'rasters', settings, profile_id=mchd.PROFILE_ID)
            self.assertEqual(native['native_adapter'], mchd.ADAPTER_ID)
            source.write_bytes(source.read_bytes().replace('ВерсФорм="EMCHD_1"'.encode(), 'ВерсФорм="other"'.encode()))
            with self.assertRaisesRegex(NativeMchdError, 'version_not_registered'):
                decode(source, root/'rasters', settings, profile_id=mchd.PROFILE_ID)


if __name__ == '__main__':
    unittest.main()
