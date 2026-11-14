"""Prepare a local reused-document regression set; never distribute source images.

Run from the repository: python -m experiments.yolo26.prepare_pilot --output ...
References are separate from the inference whitelist and never passed to a model.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

from harness_ocr.util import file_sha, write_json, utc
from harness_ocr.profiles import ProfileRegistry


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('pilot_output_must_be_new')
    identity_root = workspace / 'xiaomi-ocr-lab'
    identity_manifest = identity_root / 'benchmark/expansion_sources_identity.json'
    fixture_root = workspace / 'mosru-ocr-fixtures'
    fixture_manifest = fixture_root / 'manifest.json'
    identities = _read(identity_manifest)['cases']
    fixtures = _read(fixture_manifest)['cases']
    documents, references = [], {}
    for suffix in ('89', '91'):
        case = next(c for c in identities if c['id'] == f'midv2020_ru_{suffix}_upright')
        expected = case['expected_fields']
        number = expected['id_number']
        gold = {'holder.birth_date': expected['birth_date'],
                'passport.series': number[:4], 'passport.number': number[4:]}
        doc = {'id': case['id'], 'original_id': case['group_id'],
               'path': str((identity_root / case['image_path']).resolve()),
               'sha256': case['sha256'], 'profile_id': 'D01', 'subset': sorted(gold),
               'cohort': 'public_physical_scan_of_mock_identity', 'synthetic': False,
               'identity_values_fictional': True, 'original_split': case['split'],
               'exposure': 'reused_in_previous_project_studies_not_new_holdout',
               'source': case['provenance']}
        documents.append(doc)
        references[doc['id']] = {'gold': gold, 'gold_basis': 'existing_author_annotation',
            'optical_literals': [expected[k] for k in ('surname', 'given_name', 'patronymic', 'birth_date')],
            'unscored': ['full_name_composition', 'full_page_CER', 'all_other_profile_fields']}
    selected = [('pay_to_002', 'photo_degraded'), ('pay_deregister_002', 'clean_200dpi'),
                ('contract_paid_002', 'photo_degraded'), ('contract_budget_002', 'clean_200dpi')]
    payment_map = {'payee.name': 'payee.name', 'payee.inn': 'payee.inn',
                   'payee.kpp': 'payee.kpp', 'payer.name': 'payer.name',
                   'payer.inn': 'payer.inn', 'payment.purpose': 'payment.purpose'}
    contract_map = {'contract.number': 'contract.number', 'contract.date': 'contract.date',
                    'provider.name': 'provider.name', 'provider.inn': 'provider.inn',
                    'contract.program': 'contract.program', 'child.birth_date': 'learner.birth_date'}
    for case_id, kind in selected:
        case = next(c for c in fixtures if c['id'] == case_id)
        variant = next(v for v in case['variants'] if v['kind'] == kind and v['page'] == 1)
        gold_path = fixture_root / case['gold_path']
        if file_sha(gold_path) != case['gold_sha256']:
            raise ValueError('fixture_reference_sha_mismatch:' + case_id)
        mapping = payment_map if case['document_type'] == 'payment_notice' else contract_map
        fields = _read(gold_path)['fields']
        visible = [f for f in fields if f['printed_presence'] == 'present'
                   and isinstance(f.get('raw_value'), str) and f['raw_value']
                   and any(e.get('page') == 1 for e in f.get('evidence', []))]
        gold = {mapping[f['name']]: f['raw_value'] for f in visible if f['name'] in mapping}
        doc = {'id': variant['id'], 'original_id': case['group_id'],
               'path': str((fixture_root / variant['path']).resolve()),
               'sha256': variant['sha256'],
               'profile_id': 'D07' if case['document_type'] == 'payment_notice' else 'D09',
               'subset': sorted(gold), 'cohort': 'synthetic_mosru_seen_template_regression',
               'synthetic': True, 'original_split': case['subset'], 'variant': kind,
               'exposure': 'reused_template_and_examples_not_new_holdout',
               'source': {'dataset': 'mosru_synthetic_role_fixtures_v1',
                          'source_template_id': case['source_template_id'],
                          'original_gold_sha256': case['gold_sha256'],
                          'page_scope': 'page1_only_explicit_profile_subset'}}
        documents.append(doc)
        references[doc['id']] = {'gold': gold,
            'optical_literals': [f['raw_value'] for f in visible],
            'gold_basis': 'existing_fictional_fixture_generation_reference_page1_only',
            'unscored': ['page2_fields', 'full_page_CER', 'typed_composite_fields_without_exact_reference']}
    registry = ProfileRegistry()
    for doc in documents:
        path = Path(doc['path'])
        if not path.is_file() or file_sha(path) != doc['sha256']:
            raise ValueError('pilot_source_missing_or_changed:' + doc['id'])
        registry.compile(doc['profile_id'], subset=doc['subset'])
    output.mkdir(parents=True)
    (output / 'documents').mkdir()
    for doc in documents:
        source = Path(doc['path'])
        relative = Path('documents') / (doc['id'] + source.suffix.lower())
        target = output / relative
        shutil.copyfile(source, target)
        if file_sha(target) != doc['sha256']:
            raise ValueError('pilot_copy_sha_mismatch:' + doc['id'])
        doc['path'] = relative.as_posix()
    inference = {'schema_version': 'yolo26-inference/1', 'created_utc': utc(),
        'contains_ground_truth': False, 'dataset_status': 'reused_regression_not_new_holdout',
        'documents': documents}
    reference = {'schema_version': 'yolo26-references/1', 'created_utc': utc(),
        'scoring_only': True, 'documents': references,
        'metric': 'exact raw scalar literal without normalization; optical substring recovery is role-free diagnostic'}
    write_json(output / 'inference.json', inference)
    write_json(output / 'references.json', reference)
    audit = {'schema_version': 'yolo26-data-freeze/1', 'created_utc': utc(),
        'inference_sha256': file_sha(output / 'inference.json'),
        'references_sha256': file_sha(output / 'references.json'),
        'input_manifests': {str(p): file_sha(p) for p in (identity_manifest, fixture_manifest)},
        'documents': len(documents), 'original_groups': len({d['original_id'] for d in documents}),
        'scored_scalar_fields': sum(len(r['gold']) for r in references.values()),
        'exclusions': [
            {'population': 'cash_register_receipts', 'reason': 'no_receipt_profile_in_current_ten_profiles; separate_optics_only_corpus_required'},
            {'population': 'native_MCHD_XML', 'reason': 'optical_detector_not_applicable'},
            {'population': 'contract_page2', 'reason': 'bounded_pilot_uses_page1_only; no_full_contract_claim'},
            {'population': 'other_fields', 'reason': 'no_audited_exact_scalar_mapping_in_this_pilot'}],
        'quality_qualified': False, 'inference_executed': False,
        'new_independent_holdout': False, 'real_person_data': False}
    write_json(output / 'data-freeze.json', audit)
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.workspace, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
