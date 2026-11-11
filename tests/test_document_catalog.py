"""Guard the colleague's documented input/output contract against profile drift."""
import json
from pathlib import Path
import re
import unittest

from harness_ocr.profiles import ProfileRegistry


ROOT = Path(__file__).resolve().parents[1]


class DocumentCatalogTests(unittest.TestCase):
    def test_catalog_matches_all_ten_executable_profile_fields(self):
        catalog = (ROOT / 'docs/supported-documents.md').read_text(encoding='utf-8')
        sections = re.split(r'^### (D\d\d) — .*$', catalog, flags=re.MULTILINE)
        documented = dict(zip(sections[1::2], sections[2::2]))
        registry = ProfileRegistry()
        aliases = {item['document_id'] for item in registry.list()}
        self.assertEqual(set(documented), aliases)
        self.assertEqual(len(aliases), 10)
        total = 0
        for alias in sorted(aliases):
            with self.subTest(profile=alias):
                declaration = registry.get(alias)
                compiled = registry.compile(alias)
                section = documented[alias]
                self.assertIn('`' + compiled['profile_id'] + '`', section)
                rows = re.findall(
                    r'^\| `([^`]+)` \| .* \| (Обязательное|Условное) \|$',
                    section, flags=re.MULTILINE)
                expected = {
                    field['key']: ('Обязательное' if field['requirement'] ==
                                   'required_for_profile_completion' else 'Условное')
                    for field in declaration['fields']}
                self.assertEqual(dict(rows), expected)
                self.assertEqual(len(rows), len(expected))
                self.assertEqual(set(expected), set(compiled['field_definitions']))
                total += len(rows)
        self.assertEqual(total, 96)

    def test_own_document_example_uses_real_profile_and_no_gold(self):
        manifest = json.loads((ROOT / 'docs/own-documents.example.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['schema_version'], 'harness-own-documents/1')
        self.assertTrue(manifest['documents'])
        for document in manifest['documents']:
            self.assertNotIn('gold', document)
            self.assertFalse(Path(document['path']).is_absolute())
            ProfileRegistry().compile(document['profile_id'], subset=document.get('subset'))


if __name__ == '__main__':
    unittest.main()
