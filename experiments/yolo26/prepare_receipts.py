"""Prepare three reused public real receipts for optical-only comparison."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from harness_ocr.util import file_sha, utc, write_json


def prepare(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('pilot_output_must_be_new')
    root = workspace / 'xiaomi-ocr-lab'
    source = root / 'benchmark/expansion_receipts_test.json'
    manifest = json.loads(source.read_text(encoding='utf-8-sig'))
    documents, references = [], {}
    # Fixed existing manifest order, chosen without any current detector output.
    for case in manifest['cases'][:3]:
        path = (root / case['image_path']).resolve()
        if not path.is_file() or file_sha(path) != case['sha256']:
            raise ValueError('receipt_source_changed:' + case['id'])
        doc = {'id': case['id'], 'original_id': case['group_id'], 'path': str(path),
            'sha256': case['sha256'], 'profile_id': None, 'subset': [],
            'allowed_stages': ['optics'], 'cohort': 'public_real_receipt_photograph_publisher_processed',
            'synthetic': False, 'original_split': case['split'],
            'exposure': 'reused_previous_receipt_studies_not_new_holdout',
            'source': {'source_url': case['source_url'], 'source_repo': case['source_repo'],
                'source_revision': case['source_revision'], 'capture_note': case.get('capture_note'),
                'license_declared': manifest.get('license_declared')}}
        documents.append(doc)
        references[doc['id']] = {'gold': {}, 'optical_literals': list(case['expected_fields'].values()),
            'gold_basis': case['field_reference_basis'],
            'unscored': ['field_roles', 'full_page_CER', 'unsupported_receipt_E2E_profile']}
    output.mkdir(parents=True)
    (output / 'documents').mkdir()
    for doc in documents:
        source_path = Path(doc['path'])
        relative = Path('documents') / (doc['id'] + source_path.suffix.lower())
        target = output / relative
        shutil.copyfile(source_path, target)
        if file_sha(target) != doc['sha256']:
            raise ValueError('receipt_copy_sha_mismatch:' + doc['id'])
        doc['path'] = relative.as_posix()
    write_json(output / 'inference.json', {'schema_version': 'yolo26-inference/1',
        'created_utc': utc(), 'contains_ground_truth': False,
        'dataset_status': 'reused_optics_only_not_new_holdout', 'documents': documents})
    write_json(output / 'references.json', {'schema_version': 'yolo26-references/1',
        'created_utc': utc(), 'scoring_only': True, 'documents': references})
    freeze = {'schema_version': 'yolo26-data-freeze/1', 'created_utc': utc(),
        'inference_sha256': file_sha(output / 'inference.json'),
        'references_sha256': file_sha(output / 'references.json'),
        'input_manifests': {str(source): file_sha(source)}, 'documents': len(documents),
        'original_groups': len({d['original_id'] for d in documents}), 'scored_scalar_fields': 0,
        'optical_reference_literals': sum(len(r['optical_literals']) for r in references.values()),
        'quality_qualified': False, 'inference_executed': False, 'new_independent_holdout': False,
        'exclusions': [{'population': 'receipt_E2E', 'reason': 'no_current_receipt_profile; no_new_parser_or_validator_added'}]}
    write_json(output / 'data-freeze.json', freeze)
    return freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.workspace, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
