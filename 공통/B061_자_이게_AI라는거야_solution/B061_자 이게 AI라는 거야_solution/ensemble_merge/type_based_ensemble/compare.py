#!/usr/bin/env python3
"""Compare a completed run with a reference; no parameter fitting or output edits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {r['id']: r['answer'] for r in rows}
    if len(rows) != 6714 or len(result) != 6714 or not set(result.values()) <= set('abcd'):
        raise ValueError('Invalid submission')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--reference', type=Path,
                        default=HERE.parent.parent / '참고자료/reference_submission.csv')
    args = parser.parse_args()
    run_dir = args.run_dir or HERE / 'runs' / json.loads((HERE / 'runs/latest.json').read_text())['run_id']
    verification = json.loads((run_dir / 'verification.json').read_text())
    candidate_path = run_dir / 'submission.csv'
    before = candidate_path.read_bytes()
    candidate_hash = hashlib.sha256(before).hexdigest()
    if candidate_hash != verification['output_sha256']:
        raise ValueError('Completed output was modified')
    candidate, reference = read(candidate_path), read(args.reference)
    if list(candidate) != list(reference):
        raise ValueError('ID order differs')
    differences = [[k, a, reference[k]] for k, a in candidate.items() if a != reference[k]]
    report = {'compared_at': datetime.now(timezone.utc).isoformat(),
              'run_id': verification['run_id'], 'comparison_after_generation_only': True,
              'rows': len(candidate), 'same': len(candidate) - len(differences),
              'different': len(differences), 'candidate_sha256': candidate_hash,
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'weights_or_predictions_changed': False, 'official_test_accuracy_measured': False}
    output_dir = run_dir / 'comparison_results'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'reference_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'differences_from_reference.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['id', 'candidate', 'reference'])
        writer.writerows(differences)
    assert candidate_path.read_bytes() == before
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
