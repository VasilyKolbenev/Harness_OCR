"""Semantic token-prefix regressions from observed selector/verifier failures.

These use a real LM Format Enforcer token trie with a tiny explicit vocabulary,
not mocked generated answers or neural weights.
"""
import json
import unittest

from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from harness_ocr.inference.constraints import (
    ConstraintError, _valid, build_prefix_constraint, build_response_schema)


EOS = 128


def vocabulary(extra=None):
    mapping = {i: chr(i) for i in range(128)}
    mapping.update(extra or {})
    return mapping, TokenEnforcerTokenizerData(
        [(i, text, False) for i, text in mapping.items()],
        lambda ids: ''.join(mapping.get(i, '') for i in ids), EOS, False, 1024)


def selector_prompt():
    return json.dumps({'contract': {}, 'fields': [
        {'key': 'passport.number', 'role': 'document'},
        {'key': 'payer.inn', 'role': 'payer'}], 'candidates': [
        {'id': 'c1', 'field_key': 'passport.number', 'party_role': 'document'},
        {'id': 'c2', 'field_key': 'passport.number', 'party_role': 'document'},
        {'id': 'p1', 'field_key': 'payer.inn', 'party_role': 'payer'}],
        'regions': [{'id': 'r1'}, {'id': 'r2'}]})


def visual_prompt(ids=('r1',)):
    return json.dumps({'task': 'Blind optical capture', 'region_ids': list(ids)})


def advance(fn, text, sequence=None, mapping=None):
    sequence = [999] if sequence is None else list(sequence)
    reverse = {text: i for i, text in (mapping or {i: chr(i) for i in range(128)}).items()}
    fn(0, sequence)
    for char in text:
        token = reverse[char]
        allowed = fn(0, sequence)
        if token not in allowed:
            raise AssertionError(f'valid prefix rejected at {char!r}: {text!r}')
        sequence.append(token)
    return sequence


FIRST = '{"field_key":"passport.number","candidate_ids":["c1"],"party_role":"document"}'
PAYER = '{"field_key":"payer.inn","candidate_ids":["p1"],"party_role":"payer"}'


class SemanticPrefixTests(unittest.TestCase):
    def make(self, purpose='selector', prompt=None, extra=None):
        mapping, data = vocabulary(extra)
        request = prompt or (selector_prompt() if purpose == 'selector' else visual_prompt())
        return mapping, build_prefix_constraint(data, request, purpose=purpose)

    def test_unreadable_branch_cannot_start_a_literal_or_an_escaped_newline(self):
        _, fn = self.make('visual_read', extra={300: '\\n', 301: 'whole page text'})
        seq = advance(fn, '{"readings":{"r1":{"status":"unreadable","literal":"')
        allowed = fn(0, seq)
        self.assertIn(ord('"'), allowed)
        for token in (ord('a'), ord('\\'), 300, 301):
            self.assertNotIn(token, allowed)
        self.assertIsNone(fn.enforcer.prefix_states[tuple(seq[1:])].parser.shortcut_key())
        seq = advance(fn, '"}}}', seq)
        self.assertEqual(fn(0, seq), [EOS])

    def test_multichar_doomed_visual_branch_is_not_allowed_from_the_root(self):
        bad = '{"readings":{"r1":{"status":"unreadable","literal":"text\\n'
        good = '{"readings":{"r1":{"status":"unreadable","literal":""}}}'
        _, fn = self.make('visual_read', extra={300: bad, 301: good})
        self.assertNotIn(300, fn(0, [999]))
        self.assertIn(301, fn(0, [999]))
        self.assertEqual(fn(0, [999, 301]), [EOS])

    def test_readable_and_ambiguous_preserve_unicode_and_escaped_literals(self):
        text = 'Иванов 😀\n"Центр" \\'
        extra = {300+i: char for i, char in enumerate(sorted(set(text)-set(chr(i) for i in range(128))))}
        for status in ('readable', 'ambiguous'):
            for ascii_only in (False, True):
                mapping, fn = self.make('visual_read', extra=extra)
                answer = {'readings': {'r1': {'status': status, 'literal': text}}}
                seq = advance(fn, json.dumps(answer, ensure_ascii=ascii_only, separators=(',', ':')),
                              mapping=mapping)
                self.assertEqual(fn(0, seq), [EOS])
                self.assertTrue(_valid(answer, build_response_schema(visual_prompt(), 'visual_read')))

    def test_packed_incomplete_unicode_escape_token_is_a_valid_prefix_not_eos(self):
        # Qwen encodes the start of a Unicode escape as one real token "\\u".
        # LMFE's fast free-string cache alone drops it because json.loads of
        # that isolated fragment fails. Traverse the escape subtree, bounded
        # by the same parser, rather than disabling the fast cache globally.
        for packed in ('\\u', '\\u0', '\\u041'):
            mapping, fn = self.make('visual_read', extra={300: packed, 301: '\\q', 302: '\\u0Z'})
            seq = advance(fn, '{"readings":{"r1":{"status":"readable","literal":"')
            allowed = fn(0, seq)
            self.assertIn(300, allowed)
            self.assertNotIn(301, allowed)
            self.assertNotIn(302, allowed)
            seq.append(300)
            self.assertNotIn(EOS, fn(0, seq))
            seq = advance(fn, '\\u0418'[len(packed):] + '"}}}', seq)
            self.assertEqual(fn(0, seq), [EOS])

    def test_packed_escape_cannot_bypass_the_literal_bound(self):
        _, fn = self.make('visual_read', extra={300: '\\u', 301: '\\n'})
        seq = advance(fn, '{"readings":{"r1":{"status":"readable","literal":"' + 'a' * 2048)
        allowed = fn(0, seq)
        self.assertIn(ord('"'), allowed)
        for token in (ord('a'), ord('\\'), 300, 301):
            self.assertNotIn(token, allowed)
        seq = advance(fn, '"}}}', seq)
        self.assertEqual(fn(0, seq), [EOS])

    def test_statuses_of_different_regions_do_not_leak(self):
        _, fn = self.make('visual_read', visual_prompt(('r1', 'r2')))
        seq = advance(fn, '{"readings":{"r1":{"status":"unreadable","literal":""},'
                          '"r2":{"status":"readable","literal":"123456"}}}')
        self.assertEqual(fn(0, seq), [EOS])
        schema = build_response_schema(visual_prompt(), 'visual_read')
        self.assertFalse(_valid({'readings': {'r1': {'status': 'unreadable', 'literal': 'x'}}}, schema))
        for status in ('readable', 'ambiguous'):
            self.assertTrue(_valid({'readings': {'r1': {'status': status, 'literal': ''}}}, schema))

    def test_duplicate_field_is_pruned_before_it_becomes_a_completed_key(self):
        _, fn = self.make()
        seq = advance(fn, '{"proposals":[' + FIRST + ',{"field_key":"pa')
        allowed = fn(0, seq)
        self.assertIn(ord('y'), allowed)  # payer.inn is still unused.
        self.assertNotIn(ord('s'), allowed)  # passport.number is already used.
        seq = advance(fn, 'yer.inn","candidate_ids":["p1"],"party_role":"payer"}],'
                          '"reread_requests":[]}', seq)
        self.assertEqual(fn(0, seq), [EOS])

    def test_same_field_with_different_candidate_ids_is_invalid_not_just_duplicate_json(self):
        good = '{"proposals":[' + FIRST + ',' + PAYER + '],"reread_requests":[]}'
        bad = ('{"proposals":[' + FIRST + ',{"field_key":"passport.number",'
               '"candidate_ids":["c2"],"party_role":"document"}],"reread_requests":[]}')
        _, fn = self.make(extra={300: good, 301: bad})
        self.assertIn(300, fn(0, [999]))
        self.assertNotIn(301, fn(0, [999]))
        schema = build_response_schema(selector_prompt())
        self.assertTrue(_valid(json.loads(good), schema))
        self.assertFalse(_valid(json.loads(bad), schema))

    def test_used_field_that_prefixes_an_unused_field_cannot_close_early(self):
        material=json.loads(selector_prompt())
        material['fields'][1]={'key':'passport.number_suffix','role':'document'}
        material['candidates'][2]={'id':'p1','field_key':'passport.number_suffix','party_role':'document'}
        _,fn=self.make(prompt=json.dumps(material))
        seq=advance(fn,'{"proposals":['+FIRST+',{"field_key":"passport.number')
        self.assertIn(ord('_'),fn(0,seq))
        self.assertNotIn(ord('"'),fn(0,seq))
        seq=advance(fn,'_suffix","candidate_ids":["p1"],"party_role":"document"}],'
                       '"reread_requests":[]}',seq)
        self.assertEqual(fn(0,seq),[EOS])

    def test_unique_candidate_and_region_ids_are_pruned_in_their_own_arrays(self):
        _, fn = self.make()
        seq = advance(fn, '{"proposals":[{"field_key":"passport.number","candidate_ids":["c1","c')
        self.assertIn(ord('2'), fn(0, seq))
        self.assertNotIn(ord('1'), fn(0, seq))
        seq = advance(fn, '2"],"party_role":"document"}],"reread_requests":['
                          '{"field_key":"passport.number","region_ids":["r1","r', seq)
        self.assertIn(ord('2'), fn(0, seq))
        self.assertNotIn(ord('1'), fn(0, seq))
        seq = advance(fn, '2"],"reason_code":"critical_recheck"}]}', seq)
        self.assertEqual(fn(0, seq), [EOS])

    def test_separate_reread_requests_keep_the_existing_runtime_contract(self):
        _, fn = self.make()
        answer = {'proposals': [], 'reread_requests': [
            {'field_key': 'passport.number', 'region_ids': ['r1'], 'reason_code': 'missing'},
            {'field_key': 'passport.number', 'region_ids': ['r2'], 'reason_code': 'critical_recheck'}]}
        seq = advance(fn, json.dumps(answer, separators=(',', ':')))
        self.assertEqual(fn(0, seq), [EOS])

    def test_parser_branches_and_new_generations_do_not_share_used_fields(self):
        _, fn = self.make()
        first = advance(fn, '{"proposals":[' + FIRST + ',{"field_key":"pa')
        self.assertNotIn(ord('s'), fn(0, first))
        sibling = advance(fn, '{"proposals":[' + PAYER + ',{"field_key":"pa')
        self.assertIn(ord('s'), fn(0, sibling))
        self.assertNotIn(ord('y'), fn(0, sibling))
        _, fresh = self.make()
        seq = advance(fresh, '{"proposals":[{"field_key":"pa')
        self.assertIn(ord('s'), fresh(0, seq))
        self.assertIn(ord('y'), fresh(0, seq))

    def test_valid_complete_json_forces_eos_and_rejects_injected_padding(self):
        complete = '{"proposals":[],"reread_requests":[]}'
        _, fn = self.make(extra={300: complete})
        fn(0,[999])
        self.assertEqual(fn(0, [999, 300]), [EOS])
        with self.assertRaisesRegex(ConstraintError, 'token_after_complete_json'):
            fn(0, [999, 300, ord(' ')])
        _, incomplete = self.make('visual_read')
        seq = advance(incomplete, '{"readings":{"r1":{"status":"unreadable","literal":""}')
        self.assertNotIn(EOS, incomplete(0, seq))


if __name__ == '__main__':
    unittest.main()
