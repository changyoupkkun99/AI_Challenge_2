#!/usr/bin/env python3
"""Lock consensus and weight remaining answers using shared validation evidence."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import unicodedata
import uuid

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve(root, relative):
    parts = Path(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Expected relative input path')
    for part in parts.parts:
        found = [p for p in root.iterdir() if unicodedata.normalize('NFC', p.name)
                 == unicodedata.normalize('NFC', part)]
        if len(found) != 1:
            raise ValueError('Missing or ambiguous input: ' + relative)
        root = found[0]
    return root


def read_verified(spec):
    path = resolve(HERE, spec['path'])
    data = path.read_bytes()
    if sha(data) != spec['sha256']:
        raise ValueError('Input hash mismatch: ' + spec['path'])
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')))


def csv_data(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def classify(question, rules):
    for rule in rules:
        if any(re.search(pattern, question, flags=re.I) for pattern in rule['patterns']):
            return rule['key']
    return 'other'


def fit_weights(rows, keys, settings):
    if not rows:
        raise ValueError('No calibration rows')
    categories = [rule['key'] for rule in settings['rules']] + ['other']
    result = {}
    for key in keys:
        correct = sum(row[key] == row['answer'] for row in rows)
        global_p = (correct + 1) / (len(rows) + 2)
        detail = {'global_n': len(rows), 'global_correct': correct,
                  'global_smoothed_accuracy': global_p, 'types': {}}
        for category in ['global'] + categories:
            cohort = rows if category == 'global' else [r for r in rows if r['type'] == category]
            n = len(cohort)
            c = sum(row[key] == row['answer'] for row in cohort)
            fallback = category != 'global' and n < settings['minimum_type_rows']
            if category == 'global' or fallback:
                p = global_p
            else:
                alpha = settings['shrinkage_strength']
                p = (c + alpha * global_p) / (n + alpha)
            reliability = max(0.0, math.log(3 * p / (1 - p)))
            weight = round(settings['integer_weight_scale'] * reliability)
            detail['types'][category] = {'n': n, 'correct': c,
                'smoothed_accuracy': p, 'weight': weight,
                'uses_global_fallback': fallback}
        result[key] = detail
    return result


def vote(answers, weights, item_id, seed):
    scores = {a: sum(weights[k] for k, answer in answers.items() if answer == a) for a in 'abcd'}
    if not any(scores.values()):
        scores = dict(Counter(answers.values()))
        scores = {a: scores.get(a, 0) for a in 'abcd'}
    maximum = max(scores.values())
    ties = [a for a in 'abcd' if scores[a] == maximum]
    # Deterministic tie sampling is independent of row iteration and target answers.
    index = int(sha(f'{seed}:{item_id}:tie'.encode()), 16) % len(ties)
    return ties[index], scores, ties


def evaluate_crossfit(rows, keys, settings, seed):
    folds = settings['cross_validation_folds']
    fold_for = {r['image_group']: int(sha(f"{seed}:{r['image_group']}:fold".encode()), 16) % folds for r in rows}
    evaluated = []
    for fold in range(folds):
        train = [r for r in rows if fold_for[r['image_group']] != fold]
        test = [r for r in rows if fold_for[r['image_group']] == fold]
        if not test or not train:
            raise ValueError('Empty cross-validation fold')
        assert not ({r['image_group'] for r in train} & {r['image_group'] for r in test})
        fitted = fit_weights(train, keys, settings)
        best = max(keys, key=lambda k: (fitted[k]['global_correct'], -keys.index(k)))
        for row in test:
            answers = {key: row[key] for key in keys}
            predictions = {}
            for name, category in [('type_weighted', row['type']), ('global_weighted', 'global')]:
                weights = {k: fitted[k]['types'][category]['weight'] for k in keys}
                predictions[name] = vote(answers, weights, row['id'], seed)[0]
            predictions['equal_weighted'] = vote(answers, {k: 1 for k in keys}, row['id'], seed)[0]
            predictions['best_global_model'] = row[best]
            evaluated.append({'id': row['id'], 'fold': fold, 'type': row['type'],
                              'answer': row['answer'], **predictions})
    methods = ['type_weighted', 'global_weighted', 'equal_weighted', 'best_global_model']
    metrics = {}
    for category in ['all'] + [rule['key'] for rule in settings['rules']] + ['other']:
        cohort = evaluated if category == 'all' else [r for r in evaluated if r['type'] == category]
        metrics[category] = {name: {'n': len(cohort),
            'correct': sum(r[name] == r['answer'] for r in cohort),
            'accuracy': sum(r[name] == r['answer'] for r in cohort) / len(cohort) if cohort else None}
            for name in methods}
    return {'scope': 'Four-model vote only; ten-model consensus gate cannot be evaluated on unavailable validation predictions.',
            'checkpoint_selection_may_have_used_original_validation': True,
            'not_an_untouched_external_test': True,
            'grouped_by_image_hash': True, 'folds': folds,
            'methods': metrics, 'fold_sizes': dict(Counter(r['fold'] for r in evaluated))}, evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='Seed for validation folds and tied answers only')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'runs')
    args = parser.parse_args()
    config_data = (HERE / 'inputs.json').read_bytes()
    settings_data = (HERE / 'settings.json').read_bytes()
    config, settings = json.loads(config_data), json.loads(settings_data)
    keys = config['validated_model_keys']
    models = config['initial_models']
    predictions = {}
    expected_ids = {f'test_{i:04d}.jpg' for i in range(1, 6715)}
    for source in models:
        rows = read_verified(source)
        mapping = {row['id']: row['answer'] for row in rows}
        if len(rows) != 6714 or set(mapping) != expected_ids or not set(mapping.values()) <= set('abcd'):
            raise ValueError('Invalid model submission: ' + source['key'])
        predictions[source['key']] = mapping
    test_rows = read_verified(config['test_questions'])
    if len(test_rows) != 6714 or {r['id'] for r in test_rows} != expected_ids:
        raise ValueError('Invalid test questions')
    validation = read_verified(config['validation'])
    if len(validation) != len({r['id'] for r in validation}):
        raise ValueError('Duplicate validation IDs')
    for row in validation:
        if not row['id'].startswith('train_') or row['label_source'] != 'train_original':
            raise ValueError('Non-original validation labels are excluded')
        if row['answer'] not in 'abcd' or any(row[k] not in 'abcd' for k in keys):
            raise ValueError('Invalid validation prediction')
        row['type'] = classify(row['question'], settings['rules'])
    crossfit, oof = evaluate_crossfit(validation, keys, settings, args.seed)
    fitted = fit_weights(validation, keys, settings)
    output_rows, audit_rows = [], []
    locked = Counter()
    type_counts = Counter()
    routed_counts = Counter()
    tie_count = 0
    for row in test_rows:
        item_id = row['id']
        category = classify(row['question'], settings['rules'])
        type_counts[category] += 1
        all_answers = {key: d[item_id] for key, d in predictions.items()}
        weights = {k: fitted[k]['types'][category]['weight'] for k in keys}
        voted, scores, ties = vote({k: all_answers[k] for k in keys}, weights, item_id, args.seed)
        unanimous = len(set(all_answers.values())) == 1
        answer = next(iter(all_answers.values())) if unanimous else voted
        route = 'initial_consensus' if unanimous else 'validation_type_vote'
        locked[route] += 1
        if not unanimous:
            routed_counts[category] += 1
            tie_count += len(ties) > 1
        output_rows.append([item_id, answer])
        audit_rows.append([item_id, category, route] + [all_answers[s['key']] for s in models]
                          + [weights[k] for k in keys] + [scores[a] for a in 'abcd']
                          + ['|'.join(ties), answer])
    output = csv_data(['id', 'answer'], output_rows)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    runs = args.output_dir.resolve()
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'submission.csv').write_bytes(output)
    (run_dir / 'inputs_snapshot.json').write_bytes(config_data)
    (run_dir / 'settings_snapshot.json').write_bytes(settings_data)
    dump(run_dir / 'weights.json', fitted)
    dump(run_dir / 'cross_validation.json', crossfit)
    (run_dir / 'cross_validation_predictions.csv').write_bytes(csv_data(
        ['id', 'fold', 'type', 'answer', 'type_weighted', 'global_weighted', 'equal_weighted', 'best_global_model'],
        ([r[k] for k in ['id', 'fold', 'type', 'answer', 'type_weighted', 'global_weighted', 'equal_weighted', 'best_global_model']]
         for r in sorted(oof, key=lambda r: r['id']))))
    (run_dir / 'vote_audit.csv').write_bytes(csv_data(['id', 'type', 'route'] + [s['key'] for s in models]
        + [k + '_weight' for k in keys] + ['score_a', 'score_b', 'score_c', 'score_d', 'tied_answers', 'answer'], audit_rows))
    (run_dir / 'type_weights.csv').write_bytes(csv_data(
        ['model_key', 'type', 'n', 'correct', 'smoothed_accuracy', 'weight', 'uses_global_fallback'],
        ([key, category, d['n'], d['correct'], d['smoothed_accuracy'], d['weight'], d['uses_global_fallback']]
         for key in keys for category, d in fitted[key]['types'].items())))
    report = {'run_id': run_id, 'seed': args.seed, 'rows': len(test_rows),
        'initial_models': len(models), 'validated_voters': keys,
        'official_common_validation_rows': len(validation),
        'consensus_locked': locked['initial_consensus'],
        'type_weighted_rows': locked['validation_type_vote'],
        'test_type_counts': dict(type_counts), 'routed_type_counts': dict(routed_counts),
        'weighted_ties': tie_count, 'reference_read': False,
        'row_id_specific_overrides': 0,
        'input_snapshot_differences_from_original': config['input_snapshot_differences_from_original'],
        'historical_submissions_used_as_votes': False,
        'input_manifest_sha256': sha(config_data), 'settings_sha256': sha(settings_data),
        'script_sha256': sha(Path(__file__).read_bytes()), 'output_sha256': sha(output),
        'cross_validation_summary': crossfit['methods']['all']}
    assert (run_dir / 'submission.csv').read_bytes() == output
    dump(run_dir / 'verification.json', report)
    with (runs / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'run_id': run_id, 'seed': args.seed, 'output_sha256': sha(output),
                                 'submission': str(Path(run_id) / 'submission.csv'),
                                 'consensus_locked': locked['initial_consensus'],
                                 'status': 'complete'}, ensure_ascii=False) + '\n')
    dump(runs / 'latest.json', {'run_id': run_id, 'output_sha256': sha(output)})
    print(json.dumps({'run_dir': str(run_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
