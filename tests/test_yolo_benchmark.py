import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from harness_ocr.settings import Settings
from harness_ocr.ingest import decode
from harness_ocr.util import digest, file_sha, write_json
from experiments.yolo26.metrics import score_fields, score_optical_literals
from experiments.yolo26.run import load_dataset, run
from experiments.yolo26.run import main as benchmark_main
from experiments.yolo26.report import report


class FakeSampler:
    def __init__(self, owner, device):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def result(self):
        return {'wall_seconds': .1, 'runner_plus_worker_peak_sampled_rss_bytes': None,
                'peak_sampled_attributable_gpu_bytes': None}


class FakeOwner:
    calls = []
    active = 0
    maximum_active = 0
    def __init__(self, settings):
        self.settings = settings
        type(self).active += 1
        type(self).maximum_active = max(type(self).maximum_active, type(self).active)
    def close(self):
        type(self).active -= 1
    def call(self, op, args, **kwargs):
        type(self).calls.append((self.settings.text_detector, op, args))
        if op == 'probe':
            return {'completed': True, 'fixture': True}
        if op == 'ingest':
            return decode(args['path'], args['output'], self.settings)
        if op == 'cheap':
            value = '123456' if self.settings.text_detector == 'yolo26l' else '9123456'
            return {'observations': [{'text': value, 'completed': True}], 'regions': [], 'completed': True}
        raise AssertionError('unexpected_model_operation:' + op)


def dataset(root, count=2):
    data = root / 'data'
    data.mkdir()
    image = root / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(image)
    documents = [{'id': f'd{i}', 'original_id': 'same-original', 'path': str(image),
                  'sha256': file_sha(image), 'profile_id': 'D01', 'subset': ['passport.number'],
                  'cohort': 'fixture', 'exposure': 'fixture'} for i in range(count)]
    write_json(data / 'inference.json', {'schema_version': 'yolo26-inference/1',
        'contains_ground_truth': False, 'documents': documents})
    write_json(data / 'references.json', {'documents': {d['id']: {'gold': {'passport.number': '123456'},
        'optical_literals': ['123456']} for d in documents}})
    write_json(data / 'data-freeze.json', {'inference_sha256': file_sha(data / 'inference.json'),
        'references_sha256': file_sha(data / 'references.json')})
    return data


class ScoreTests(unittest.TestCase):
    def test_exact_raw_and_acceptance_are_distinct(self):
        result = {'completed': True, 'decision': 'review', 'fields': {
            'x': {'raw_value': '123 456', 'extraction_status': 'observed', 'validation_status': 'pass'},
            'y': {'raw_value': 'correct', 'extraction_status': 'ambiguous'}}}
        scored = score_fields(result, {'x': '123456', 'y': 'correct'})
        self.assertEqual(scored['correct_literal'], 0)
        self.assertEqual(scored['wrong_literal'], 1)
        self.assertEqual(scored['withheld'], 1)
        self.assertEqual(scored['wrong_validation_pass'], 1)
        self.assertEqual(scored['wrong_accepted'], 0)
        result['decision'] = 'accept_for_autofill'
        self.assertEqual(score_fields(result, {'x': '123456'})['wrong_accepted'], 0)
        result['fields']['x']['decision'] = 'accept_for_autofill'
        self.assertEqual(score_fields(result, {'x': '123456'})['wrong_accepted'], 1)
        result['fields']['x']['validation_status'] = 'fail'
        self.assertEqual(score_fields(result, {'x': '123456'})['wrong_accepted'], 0)
        result['fields']['x']['validation_status'] = 'pass'
        result['fields']['x']['needs_review'] = True
        self.assertEqual(score_fields(result, {'x': '123456'})['wrong_accepted'], 0)

    def test_optical_score_has_numeric_boundaries_and_no_line_concatenation(self):
        observations = [{'text': '9123456', 'completed': True}, {'text': 'Альфа', 'completed': True},
                        {'text': 'Бета', 'completed': True}, {'text': '999', 'completed': False}]
        scored = score_optical_literals(observations, ['123456', 'Альфа Бета', 'Альфа', '999'])
        self.assertEqual(scored['recovered'], 1)
        self.assertIn('not_field_accuracy', scored['scope'])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        FakeOwner.calls = []
        FakeOwner.active = FakeOwner.maximum_active = 0

    def test_cli_preserves_environment_attention_and_default_input_limit(self):
        base = Settings(Path('state'), Path('models'), device='cpu', dtype='float32', attention_backend='sdpa')
        argv = ['benchmark', '--data', 'data', '--output', 'output', '--models', 'models', '--device', 'cpu']
        with patch('sys.argv', argv), patch('sys.stdout', new_callable=io.StringIO), \
                patch('experiments.yolo26.run.Settings.from_env', return_value=base), \
                patch('experiments.yolo26.run.run', return_value={}) as run_mock:
            benchmark_main()
        settings = run_mock.call_args.args[2]
        self.assertEqual(settings.attention_backend, 'sdpa')
        self.assertEqual(settings.max_input_tokens, 8192)
        self.assertEqual(settings.max_new_tokens, 1024)

    def test_cli_explicit_common_v2_attention_context_and_timeouts(self):
        base = Settings(Path('state'), Path('models'), device='cpu', dtype='float32')
        argv = ['benchmark', '--data', 'data', '--output', 'output', '--models', 'models', '--device', 'cpu',
                '--max-input-tokens', '32768', '--attention-backend', 'sdpa', '--call-timeout', '600',
                '--document-timeout', '1800']
        with patch('sys.argv', argv), patch('sys.stdout', new_callable=io.StringIO), \
                patch('experiments.yolo26.run.Settings.from_env', return_value=base), \
                patch('experiments.yolo26.run.run', return_value={}) as run_mock:
            benchmark_main()
        settings = run_mock.call_args.args[2]
        self.assertEqual((settings.attention_backend, settings.max_input_tokens), ('sdpa', 32768))
        self.assertEqual((settings.call_timeout_seconds, settings.document_timeout_seconds), (600, 1800))
        self.assertEqual(settings.max_new_tokens, 1024)

    def test_serial_independent_optics_and_reference_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root)
            settings = Settings(root / 'state', root / 'models', device='cpu', dtype='float32')
            with patch('experiments.yolo26.run.code_pins', return_value={'fixture': 'same'}):
                summary = run(data, root / 'run', settings, stage='optics', fixture=True,
                              owner_factory=FakeOwner, sampler_factory=FakeSampler)
            self.assertEqual(summary['planned_rows'], 6)
            self.assertEqual(FakeOwner.maximum_active, 1)
            self.assertEqual(FakeOwner.active, 0)
            self.assertEqual(sum(op == 'cheap' for _, op, _ in FakeOwner.calls), 6)
            self.assertEqual({op for _, op, _ in FakeOwner.calls}, {'probe', 'ingest', 'cheap'})
            for _, _, args in FakeOwner.calls:
                self.assertNotIn('123456', json.dumps(args))
                self.assertNotIn('gold', args)
            by_arm = {m['arm']: m for m in summary['metrics']}
            self.assertEqual(by_arm['db']['literal_recovered'], 0)
            self.assertEqual(by_arm['yolo26l']['literal_recovered'], 2)
            self.assertEqual(by_arm['yolo26l']['original_groups_completed'], 1)
            self.assertIsNone(by_arm['yolo26l']['max_sampled_attributable_gpu_bytes'])
            self.assertTrue(summary['fixture'])
            self.assertFalse(summary['quality_qualified'])
            self.assertTrue((root / 'run/report.html').is_file())

    def test_reference_mutation_and_source_mutation_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root)
            refs = data / 'references.json'
            refs.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'data_freeze_changed'):
                load_dataset(data)
            self.assertFalse(FakeOwner.calls)

    def test_relative_pack_paths_work_and_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root, 1)
            inference = json.loads((data / 'inference.json').read_text())
            packed = data / 'source.png'
            packed.write_bytes((root / 'source.png').read_bytes())
            inference['documents'][0]['path'] = 'source.png'
            write_json(data / 'inference.json', inference)
            freeze = json.loads((data / 'data-freeze.json').read_text())
            freeze['inference_sha256'] = file_sha(data / 'inference.json')
            write_json(data / 'data-freeze.json', freeze)
            loaded, _ = load_dataset(data)
            self.assertEqual(Path(loaded['documents'][0]['path']), packed.resolve())
            packed.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'source_changed'):
                load_dataset(data)
            self.assertFalse(FakeOwner.calls)

    def test_optics_only_receipt_cannot_enter_e2e(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root, 1)
            inference = json.loads((data / 'inference.json').read_text())
            inference['documents'][0]['allowed_stages'] = ['optics']
            write_json(data / 'inference.json', inference)
            freeze = json.loads((data / 'data-freeze.json').read_text())
            freeze['inference_sha256'] = file_sha(data / 'inference.json')
            write_json(data / 'data-freeze.json', freeze)
            with self.assertRaisesRegex(ValueError, 'document_not_supported_in_selected_stage'):
                run(data, root / 'run', Settings(root / 'state', root / 'models', device='cpu', dtype='float32'),
                    stage='e2e', fixture=True, owner_factory=FakeOwner, sampler_factory=FakeSampler)
            self.assertFalse(FakeOwner.calls)

    def test_missing_rows_are_not_zero_quality_and_reference_sha_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root, 1)
            settings = Settings(root / 'state', root / 'models', device='cpu', dtype='float32')
            with patch('experiments.yolo26.run.code_pins', return_value={'fixture': 'same'}):
                summary = run(data, root / 'run', settings, stage='optics', fixture=True,
                              owner_factory=FakeOwner, sampler_factory=FakeSampler)
            measurements = root / 'run/measurements.json'
            measured = json.loads(measurements.read_text())
            write_json(measurements, [r for r in measured if r['arm'] != 'yolo26x'])
            updated = report(root / 'run', data / 'references.json')
            missing = next(m for m in updated['metrics'] if m['arm'] == 'yolo26x')
            self.assertEqual(missing['missing'], 1)
            self.assertEqual(missing['completed'], 0)
            self.assertIsNone(missing['median_wall_seconds_completed'])
            contrast = next(c for c in updated['paired_contrasts'] if c['arm_minus_db'] == 'yolo26x')
            self.assertIsNone(contrast['delta_correct_or_recovered'])
            (data / 'references.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'reference_changed_after_freeze'):
                report(root / 'run', data / 'references.json')

    def test_partial_e2e_values_remain_separate_from_completed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dataset(root, 1)
            settings = Settings(root / 'state', root / 'models', device='cpu', dtype='float32')
            with patch('experiments.yolo26.run.code_pins', return_value={'fixture': 'same'}):
                run(data, root / 'run', settings, stage='optics', fixture=True,
                    owner_factory=FakeOwner, sampler_factory=FakeSampler)
            protocol_path = root / 'run/protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['stage'] = 'e2e'
            protocol['protocol_sha256'] = digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})
            write_json(protocol_path, protocol)
            measurements_path = root / 'run/measurements.json'
            measurements = json.loads(measurements_path.read_text())
            edited = next(r for r in measurements if r['arm'] == 'yolo26l')
            edited['status'] = 'incomplete'
            result_path = root / 'run' / edited['result_path']
            result = json.loads(result_path.read_text())
            result.update(completed=False, decision='review', fields={'passport.number': {
                'raw_value': '123456', 'extraction_status': 'observed', 'validation_status': 'pass'}})
            write_json(result_path, result)
            write_json(measurements_path, measurements)
            summary = report(root / 'run', data / 'references.json')
            metric = next(m for m in summary['metrics'] if m['arm'] == 'yolo26l')
            self.assertEqual(metric['completed'], 0)
            self.assertEqual(metric['incomplete'], 1)
            self.assertEqual(metric['total'], 0)
            self.assertIsNone(metric['exact_literal_fraction'])
            self.assertEqual(metric['partial_result_counts_separate_from_completed']['correct_literal'], 1)
            contrast = next(c for c in summary['paired_contrasts'] if c['arm_minus_db'] == 'yolo26l')
            self.assertEqual(contrast['matched_completed_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
