#!/usr/bin/env python3
"""Merge fixed CSV inputs; never read reference or apply per-row corrections."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def locate(root, relative):
    path = root
    parts = Path(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Expected a relative package path')
    for part in parts.parts:
        found = [p for p in path.iterdir() if unicodedata.normalize('NFC', p.name)
                 == unicodedata.normalize('NFC', part)]
        if len(found) != 1:
            raise ValueError('Missing or ambiguous path: ' + relative)
        path = found[0]
    return path


def load_prediction(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != ['id', 'answer']:
        raise ValueError('Unexpected CSV columns')
    rows = list(reader)
    mapping = {r['id']: r['answer'] for r in rows}
    expected = {f'test_{i:04d}.jpg' for i in range(1, 6715)}
    if len(rows) != 6714 or len(mapping) != 6714 or set(mapping) != expected:
        raise ValueError('Unexpected or duplicate IDs')
    if not set(mapping.values()) <= set('abcd'):
        raise ValueError('Invalid answer values')
    return [r['id'] for r in rows], mapping


def encode_csv(header, rows):
    buf = io.StringIO(newline='')
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    config_bytes = (Path(__file__).parent / 'merge_config.json').read_bytes()
    config = json.loads(config_bytes)
    sources = config['sources']
    if config['tie_order'] != list('abcd'):
        raise ValueError('Unexpected tie order')
    group_sizes = {g: sum(s['group'] == g for s in sources)
                   for g in ['initial_models', 'historical_results']}
    if group_sizes != {'initial_models': 10, 'historical_results': 46}:
        raise ValueError('Input pool differs from frozen policy')
    expected_weight = {'initial_models': 23, 'historical_results': 5}
    ids = None
    predictions = []
    signatures = set()
    for source in sources:
        data = locate(ROOT, source['path']).read_bytes()
        if digest(data) != source['sha256']:
            raise ValueError('Input changed: ' + source['path'])
        if source['weight'] != expected_weight[source['group']]:
            raise ValueError('Weight differs from frozen group-balanced policy')
        row_ids, prediction = load_prediction(data)
        if ids is None:
            ids = row_ids
        elif ids != row_ids:
            raise ValueError('Input row order differs: ' + source['path'])
        signature = digest(''.join(prediction[k] for k in sorted(prediction)).encode())
        if signature != source['answer_sha256'] or signature in signatures:
            raise ValueError('Duplicate or changed answer vector')
        signatures.add(signature)
        predictions.append(prediction)
    merged = []
    audit = []
    ties = 0
    for item_id in ids:
        scores = {a: 0 for a in 'abcd'}
        for source, prediction in zip(sources, predictions):
            scores[prediction[item_id]] += source['weight']
        maximum = max(scores.values())
        tied = [a for a in 'abcd' if scores[a] == maximum]
        answer = tied[0]
        ties += len(tied) > 1
        merged.append([item_id, answer])
        audit.append([item_id] + [p[item_id] for p in predictions]
                     + [scores[a] for a in 'abcd'] + [answer, '|'.join(tied)])
    output = encode_csv(['id', 'answer'], merged)
    audit_data = encode_csv(['id'] + [s['key'] for s in sources]
                            + ['score_a', 'score_b', 'score_c', 'score_d',
                               'merged_answer', 'tied_answers'], audit)
    report = {
        'rows': len(ids), 'source_count': len(sources),
        'group_sizes': group_sizes,
        'group_total_weights': {'initial_models': 230, 'historical_results': 230},
        'weight_per_initial_model': 23, 'weight_per_historical_csv': 5,
        'tie_order': list('abcd'), 'ties': ties,
        'row_level_overrides': 0,
        'inputs_hash_verified': len(sources),
        'answer_0047': dict(merged)['test_0047.jpg'],
        'output_sha256': digest(output),
        'config_sha256': digest(config_bytes),
        'audit_sha256': digest(audit_data),
        'official_accuracy_measured': False,
    }
    report_data = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    destination = args.output_dir.resolve()
    outputs = [(destination / 'submission_ensemble_merge.csv', output),
               (destination / 'ensemble_merge/results/vote_audit.csv', audit_data),
               (destination / 'ensemble_merge/results/verification.json', report_data)]
    for path, data in outputs:
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Different output exists; choose another --output-dir: ' + str(path))
    for path, data in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise ValueError('Output readback mismatch')
    with (destination / 'ensemble_merge/results/execution_log.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps({'run_at': datetime.now(timezone.utc).isoformat(),
                              'config_sha256': digest(config_bytes),
                              'output_sha256': digest(output),
                              'event': 'merge_completed_without_reference'}, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
