"""Published normative examples and adversarial arithmetic contracts; no NN."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness_ocr import bank_keys


class BankKeyTests(unittest.TestCase):
    def test_registry_is_pinned_and_sources_are_explicit_legal_reproductions(self):
        registry = bank_keys.source_registry()
        self.assertEqual(hashlib.sha256(bank_keys.REGISTRY_PATH.read_bytes()).hexdigest(), bank_keys.REGISTRY_SHA256)
        self.assertEqual(registry['weights'], [7, 1, 3] * 7 + [7, 1])
        self.assertEqual(len(registry['sources']), 3)
        self.assertFalse(registry['current_legal_applicability_verified'])
        self.assertIn('legal reproductions', registry['source_type'])
        self.assertEqual(registry['validation_scope'], 'arithmetic_consistency_only')

    def test_three_supported_published_examples_and_fourth_clearing_family(self):
        cases = bank_keys.source_registry()['published_examples']
        self.assertEqual(len(cases), 4)
        for case in cases:
            with self.subTest(example=case['number']):
                result = bank_keys.validate_bank_key(case['account_literal'], case['bik'], location=case['location'])
                self.assertEqual(result['validation_status'], 'pass' if case['supported_ascii_contract'] else 'not_checked')
                self.assertEqual(result['transform_ids'], [])
                self.assertFalse(result['normalization_performed'])
                self.assertFalse(result['automatic_acceptance_authority'])
                self.assertFalse(result['production_qualified'])
        self.assertIn('clearing_currency_literal_not_supported', result['validation_reasons'])

    def test_rkc_client_is_not_the_credit_institution_rule(self):
        # Published example 2: it is a client account, but opened in an RKC.
        account, bik = '40102810100000010001', '040305000'
        self.assertEqual(bank_keys.validate_bank_key(account, bik, location='bank_of_russia_unit')['validation_status'], 'pass')
        self.assertEqual(bank_keys.validate_bank_key(account, bik, location='credit_institution')['validation_status'], 'fail')
        self.assertEqual(bank_keys.validate_bank_key(account, bik)['validation_status'], 'not_checked')

    def test_missing_location_does_not_choose_whichever_prefix_matches(self):
        for account, bik in [('30101810800000000746', '049805000'), ('40602810700000000025', '049805746')]:
            result = bank_keys.validate_bank_key(account, bik)
            self.assertEqual(result['validation_status'], 'not_checked')
            self.assertEqual(result['validation_reasons'], ['account_opening_location_not_established'])
        for location in ('settlement', 'correspondent', True, [], {}, 'credit_institution '):
            with self.subTest(location=location):
                result = bank_keys.validate_bank_key('40602810700000000025', '049805746', location=location)
                self.assertEqual(result['validation_status'], 'not_checked')

    def test_each_single_account_digit_mutation_fails_without_repair(self):
        original = '40602810700000000025'
        for index, digit in enumerate(original):
            changed = original[:index] + str((int(digit) + 1) % 10) + original[index+1:]
            with self.subTest(position=index+1):
                result = bank_keys.validate_bank_key(changed, '049805746', location='credit_institution')
                self.assertEqual(result['validation_status'], 'fail')
                self.assertEqual(result['transform_ids'], [])
                self.assertNotIn('canonical_value', result)
                self.assertNotIn('expected_key', result)

    def test_checksum_collision_is_not_optical_or_ownership_proof(self):
        # Positions 16 and 19 have the same weight. This is a distinct literal,
        # not a transcription repair, and both satisfy the arithmetic contract.
        values = ('40602810700000000025', '40602810700000020005')
        for literal in values:
            result = bank_keys.validate_bank_key(literal, '049805746', location='credit_institution')
            self.assertEqual(result['validation_status'], 'pass')
            self.assertEqual(result['validation_scope'], 'arithmetic_consistency_only')
            self.assertFalse(result['automatic_acceptance_authority'])

    def test_literal_formats_are_never_cleaned_or_confusable_replaced(self):
        for account, bik in [('406028107 00000000025', '049805746'), ('4060281070000000002О', '049805746'),
                             ('٤٠٦٠٢٨١٠٧٠٠٠٠٠٠٠٠٠٢٥', '049805746'), ('40602810700000000025', '049 805746'),
                             ('40602810700000000025', '٠٤٩٨٠٥٧٤٦'), ('0'*20, '049805746'),
                             ('40602810700000000025', '0'*9), (123, '049805746'),
                             ('40602810700000000025', None)]:
            with self.subTest(account=account, bik=bik):
                result = bank_keys.validate_bank_key(account, bik, location='credit_institution')
                self.assertEqual(result['validation_status'], 'fail')
                self.assertFalse(result['normalization_performed'])

    def test_treasury_like_identifier_does_not_receive_a_bank_key(self):
        result = bank_keys.validate_bank_key('03100643000000019500', '024501901', location='bank_of_russia_unit')
        self.assertEqual(result['validation_status'], 'not_checked')
        self.assertIn('zero_prefixed_account_outside_registered_bank_scope', result['validation_reasons'])

    def test_source_tampering_or_missing_registry_cannot_give_a_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'registry.json';path.write_bytes(bank_keys.REGISTRY_PATH.read_bytes()+b' ')
            for target in (path, path.with_name('missing.json')):
                with patch.object(bank_keys, 'REGISTRY_PATH', target):
                    result = bank_keys.validate_bank_key('40602810700000000025', '049805746', location='credit_institution')
                    self.assertEqual(result['validation_status'], 'not_checked')
                    self.assertIn('bank_key_source_registry_unavailable_or_tampered', result['validation_reasons'])

    def test_per_check_output_is_closed_to_the_arithmetic_scope(self):
        result = bank_keys.validate_bank_key('40602810700000000025', '049805746', location='credit_institution')
        check = result['checks'][0]
        self.assertEqual(check, {'check_id':bank_keys.ALGORITHM_ID, 'status':'pass',
            'scope':'arithmetic_consistency_only', 'reason_codes':[], 'source_ids':list(bank_keys.SOURCE_IDS)})
        copied = bank_keys.source_registry();copied['weights'][0]=99
        self.assertEqual(bank_keys.source_registry()['weights'][0], 7)


if __name__ == '__main__':unittest.main()
