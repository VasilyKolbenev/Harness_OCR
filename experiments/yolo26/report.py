"""Post-run scorer. This is the first component that opens gold references."""
from __future__ import annotations
import argparse
import html
import json
import math
from pathlib import Path
import statistics

from harness_ocr.util import digest, file_sha, utc, write_json
from .metrics import score_fields, score_optical_literals


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _median(values):
    values = [x for x in values if isinstance(x, (float, int)) and math.isfinite(x)]
    return statistics.median(values) if values else None


def report(output, references):
    output, references = Path(output).resolve(), Path(references).resolve()
    protocol = read(output / 'protocol.json')
    payload = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if digest(payload) != protocol['protocol_sha256']:
        raise ValueError('protocol_sha_mismatch')
    if file_sha(references) != protocol['references_sha256']:
        raise ValueError('reference_changed_after_freeze')
    refs = read(references)['documents']
    measurements = read(output / 'measurements.json') if (output / 'measurements.json').exists() else []
    indexed = {(r['document_id'], r['arm'], r['repeat']): r for r in measurements}
    if len(indexed) != len(measurements):
        raise ValueError('duplicate_measurement_rows')
    rows = []
    for planned in protocol['planned_rows']:
        key = (planned['document_id'], planned['arm'], planned['repeat'])
        row = {**planned, **indexed.get(key, {'status': 'missing'})}
        if row.get('result_path'):
            result_path = (output / row['result_path']).resolve()
            if output not in result_path.parents:
                raise ValueError('result_path_outside_run')
            if result_path.is_file():
                result = read(result_path)
                if result.get('source_sha256') != row['source_sha256']:
                    raise ValueError('result_source_mismatch')
                ref = refs[row['document_id']]
                if protocol['stage'] == 'e2e':
                    row['score'] = score_fields(result, ref['gold'])
                else:
                    row['score'] = score_optical_literals(result.get('observations', []), ref['optical_literals'])
                row['result_sha256'] = file_sha(result_path)
        rows.append(row)
    metrics = []
    for arm in protocol['arm_order']:
        for cohort in sorted({r['cohort'] for r in rows}):
            planned = [r for r in rows if r['arm'] == arm and r['cohort'] == cohort]
            complete = [r for r in planned if r['status'] == 'completed' and 'score' in r]
            partial = [r for r in planned if r['status'] == 'incomplete' and 'score' in r]
            resources = [r.get('resources', {}) for r in complete]
            all_resources = [r.get('resources', {}) for r in planned if r['status'] != 'missing']
            warm_resources = [r.get('resources', {}) for r in planned
                              if r['status'] != 'missing' and r.get('first_timed_document_in_block') is False]
            item = {'arm': arm, 'cohort': cohort, 'planned': len(planned), 'completed': len(complete),
                'original_groups_completed': len({r['original_id'] for r in complete}),
                'incomplete': sum(r['status'] == 'incomplete' for r in planned),
                'failed': sum(r['status'] == 'failed' for r in planned),
                'missing': sum(r['status'] == 'missing' for r in planned),
                'median_wall_seconds_completed': _median([r.get('wall_seconds') for r in resources]),
                'median_wall_seconds_all_attempts': _median([r.get('wall_seconds') for r in all_resources]),
                'median_wall_seconds_all_attempts_first_excluded': _median([r.get('wall_seconds') for r in warm_resources]),
                'resource_measurement_scope': 'all_attempts_including_partial_and_failed_not_only_completed',
                'max_sampled_runner_worker_rss_bytes': max([r['runner_plus_worker_peak_sampled_rss_bytes']
                    for r in all_resources if r.get('runner_plus_worker_peak_sampled_rss_bytes') is not None], default=None),
                'max_sampled_attributable_gpu_bytes': max([r['peak_sampled_attributable_gpu_bytes']
                    for r in all_resources if r.get('peak_sampled_attributable_gpu_bytes') is not None], default=None)}
            if protocol['stage'] == 'e2e':
                for name in ('total', 'correct_literal', 'wrong_literal', 'withheld',
                             'validation_pass', 'wrong_validation_pass', 'accepted_for_autofill', 'wrong_accepted'):
                    item[name] = sum(r['score'][name] for r in complete)
                item['exact_literal_fraction'] = item['correct_literal'] / item['total'] if item['total'] else None
                item['wrong_accepted_fraction_among_accepted_scored_fields'] = (item['wrong_accepted'] / item['accepted_for_autofill']
                    if item['accepted_for_autofill'] else None)
                item['document_accept_for_autofill'] = sum(r['score']['document_decision'] == 'accept_for_autofill' for r in complete)
                item['document_review'] = sum(r['score']['document_decision'] == 'review' for r in complete)
                item['partial_result_counts_separate_from_completed'] = {
                    'documents': len(partial), **{name: sum(r['score'][name] for r in partial)
                    for name in ('total', 'correct_literal', 'wrong_literal', 'withheld',
                                 'wrong_validation_pass', 'accepted_for_autofill', 'wrong_accepted')}}
            else:
                item['literal_total'] = sum(r['score']['total'] for r in complete)
                item['literal_recovered'] = sum(r['score']['recovered'] for r in complete)
                item['partial_result_counts_separate_from_completed'] = {'documents': len(partial),
                    'literal_total': sum(r['score']['total'] for r in partial),
                    'literal_recovered': sum(r['score']['recovered'] for r in partial)}
            metrics.append(item)
    contrasts = []
    for arm in protocol['arm_order']:
        if arm == 'db':
            continue
        for cohort in sorted({r['cohort'] for r in rows}):
            pairs = []
            for row in rows:
                if row['arm'] != arm or row['cohort'] != cohort:
                    continue
                baseline = next((r for r in rows if r['arm'] == 'db' and r['document_id'] == row['document_id']
                                 and r['repeat'] == row['repeat']), None)
                if baseline and row['status'] == baseline['status'] == 'completed' and 'score' in row and 'score' in baseline:
                    key = 'correct_literal' if protocol['stage'] == 'e2e' else 'recovered'
                    a_time = row.get('resources', {}).get('wall_seconds')
                    b_time = baseline.get('resources', {}).get('wall_seconds')
                    pairs.append({'document_id': row['document_id'], 'original_id': row['original_id'],
                        'repeat': row['repeat'], 'delta_correct_or_recovered': row['score'][key] - baseline['score'][key],
                        'delta_wall_seconds': a_time - b_time if a_time is not None and b_time is not None else None})
            contrasts.append({'arm_minus_db': arm, 'cohort': cohort, 'matched_completed_pairs': len(pairs),
                'original_groups': len({p['original_id'] for p in pairs}),
                'delta_correct_or_recovered': sum(p['delta_correct_or_recovered'] for p in pairs) if pairs else None,
                'pairs': pairs, 'inference': 'descriptive_only_no_token_or_crop_pseudoreplication'})
    summary = {'schema_version': 'yolo26-summary/1', 'created_utc': utc(),
        'protocol_sha256': protocol['protocol_sha256'], 'stage': protocol['stage'],
        'fixture': protocol.get('fixture', False), 'planned_rows': len(rows),
        'status': read(output / 'status.json') if (output / 'status.json').exists() else {'status': 'unknown'},
        'metrics': metrics, 'paired_contrasts': contrasts, 'rows': rows,
        'quality_qualified': False, 'automatic_winner': None,
        'limits': ['reused_examples_not_new_independent_holdout', 'small_sample',
            'synthetic_results_not_real_world_generalization', 'partial_scalar_profile_subsets_only',
            'optics_stage_does_not_execute_Xiaomi_Qwen_or_measure_field_roles',
            'incomplete_and_missing_excluded_from_completed_quality_denominators_and_shown_separately',
            'null_GPU_measurements_not_zero_VRAM', 'CPU_speed_not_extrapolated_to_GPU',
            'arm_major_order_and_cold_first_document_limit_latency_comparability',
            'no_CER_without_full_page_reference', 'no_production_false_accept_rate_claim_from_small_pilot']}
    write_json(output / 'summary.json', summary)
    e = html.escape
    def val(x):
        return 'не измерено' if x is None else e(str(round(x, 3) if isinstance(x, float) else x))
    table_rows = []
    for item in metrics:
        quality = (f"{item['correct_literal']}/{item['total']}" if protocol['stage'] == 'e2e'
                   else f"{item['literal_recovered']}/{item['literal_total']}")
        table_rows.append('<tr>' + ''.join(f'<td>{x}</td>' for x in [e(item['arm']), e(item['cohort']),
            f"{item['completed']}/{item['planned']}", quality,
            e(str(item['partial_result_counts_separate_from_completed'])), val(item['median_wall_seconds_completed']),
            val(item['max_sampled_runner_worker_rss_bytes']), val(item['max_sampled_attributable_gpu_bytes'])]) + '</tr>')
    planned_rows = []
    for row in rows:
        link = (f'<a href="{e(row["result_path"])}">result.json</a>'
                if row.get('result_sha256') else e(row.get('error', '—')))
        planned_rows.append('<tr>' + ''.join(f'<td>{e(str(row[k]))}</td>' for k in ('document_id', 'arm', 'repeat', 'status'))
                            + f'<td>{link}</td></tr>')
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Harness OCR: DB / YOLO26</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#172331}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd3dc;padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#eaf0f8}}code{{overflow-wrap:anywhere}}.note{{padding:16px;background:#fff4d7}}</style>
<h1>Harness OCR: DB / YOLO26-L / YOLO26-X</h1>
<p>Стадия: <b>{e(protocol['stage'])}</b>. Устройство: <b>{e(protocol['settings']['device'])}</b>.
План: {len(rows)} запусков. Протокол: <code>{e(protocol['protocol_sha256'])}</code>.</p>
<p class="note">Это повторное регрессионное исследование на ранее использованных документах. Физические сканы макетов паспортов и синтетические документы mos.ru показаны отдельно. Качество на новых документах и готовность к производственной эксплуатации этим пилотом не доказаны.</p>
<p>В E2E сравниваются точные исходные значения выбранных скалярных полей, без исправления по эталону. В optics — только наличие точных эталонных фрагментов в CTC-чтении, без оценки ролей полей; Xiaomi и Qwen в этой стадии не запускаются. Неполные, ошибочные и ненаблюдавшиеся запуски не превращаются в нулевую точность.</p>
<table><thead><tr><th>Детектор</th><th>Группа</th><th>Завершено / план</th><th>Верно / эталонов завершённых</th><th>Частичные результаты отдельно</th><th>Медиана полных, с</th><th>Пик RSS всех попыток, байт</th><th>GPU всех попыток, байт</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p>RSS — выборочный максимум суммы памяти управляющего процесса и владельца моделей; общие страницы могут учитываться дважды. GPU — только память соответствующих PID по NVML. Отсутствие измерения обозначено явно. Первый документ блока может включать ленивую инициализацию; загрузка E2E вынесена в <a href="loads.json">loads.json</a>. Порядок блоков фиксирован и может влиять на скорость.</p>
<p>Строгие проверки и политика допуска сохранены. Прошедшая валидация не равна разрешению автозаполнения. Ошибочно принятые значения и число принятых полей доступны в <a href="summary.json">summary.json</a>; нулевой знаменатель не означает доказанную безошибочность.</p>
<h2>Все запланированные запуски</h2><table><thead><tr><th>Документ</th><th>Детектор</th><th>Повтор</th><th>Статус</th><th>Данные</th></tr></thead><tbody>{''.join(planned_rows)}</tbody></table>
<p><a href="protocol.json">Замороженный протокол</a> · <a href="measurements.json">Измерения</a> · <a href="summary.json">Оценки и парные разности</a> · <a href="status.json">Состояние</a></p>
<p>Нового независимого тестового набора, полной транскрипции для CER, GPU-замеров на CPU и статистически доказанного производственного риска здесь нет. Автоматического победителя отчёт не выбирает.</p></html>'''
    (output / 'report.html').write_text(page, encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(report(args.output, args.references), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
