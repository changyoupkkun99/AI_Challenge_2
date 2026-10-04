#!/usr/bin/env python3
"""Compare the frozen merge with reference after generation; never change predictions."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from merge import ROOT, digest, load_prediction, encode_csv, locate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / '참고자료/reference_submission.csv')
    parser.add_argument('--candidate', type=Path, default=ROOT / 'submission_ensemble_merge.csv')
    args = parser.parse_args()
    folder = Path(__file__).parent
    config_bytes = (folder / 'merge_config.json').read_bytes()
    frozen = json.loads((folder / 'policy_frozen_before_comparison.json').read_text(encoding='utf-8'))
    if digest(config_bytes) != frozen['config_sha256']:
        raise ValueError('Merge policy changed after freeze')
    if digest((folder / 'merge.py').read_bytes()) != frozen['merge_code_sha256']:
        raise ValueError('Merge code changed after freeze')
    verified = json.loads((folder / 'results/verification.json').read_text(encoding='utf-8'))
    candidate_bytes = args.candidate.read_bytes()
    if digest(candidate_bytes) != verified['output_sha256']:
        raise ValueError('Candidate differs from completed merge')
    reference_bytes = args.reference.read_bytes()
    ids, candidate = load_prediction(candidate_bytes)
    reference_ids, reference = load_prediction(reference_bytes)
    if ids != reference_ids:
        raise ValueError('Row order differs')
    config = json.loads(config_bytes)
    exact_reference_inputs = []
    for source in config['sources']:
        data = locate(ROOT, source['path']).read_bytes()
        if digest(data) != source['sha256']:
            raise ValueError('Source changed')
        _, answers = load_prediction(data)
        if answers == reference:
            exact_reference_inputs.append(source['key'])
    differences = [[k, candidate[k], reference[k]] for k in ids if candidate[k] != reference[k]]
    report = {'compared_at': datetime.now(timezone.utc).isoformat(),
              'comparison_only_no_refitting': True,
              'rows': len(ids), 'same': len(ids) - len(differences),
              'different': len(differences),
              'agreement': (len(ids) - len(differences)) / len(ids),
              'candidate_sha256': digest(candidate_bytes),
              'reference_sha256': digest(reference_bytes),
              'frozen_config_sha256': digest(config_bytes),
              'reference_answer_identical_inputs': exact_reference_inputs,
              'official_accuracy_measured': False}
    output = folder / 'comparison_results'
    output.mkdir(exist_ok=True)
    (output / 'reference_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'differences_from_reference.csv').write_bytes(encode_csv(['id', 'merged_answer', 'reference_answer'], differences))
    if args.candidate.read_bytes() != candidate_bytes:
        raise ValueError('Candidate changed during comparison')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
