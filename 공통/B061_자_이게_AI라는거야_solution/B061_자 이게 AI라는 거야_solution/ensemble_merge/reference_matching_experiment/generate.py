#!/usr/bin/env python3
"""Generate from saved input CSVs and learned selector, without reading a reference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from common import HERE, load_inputs, sha, dump, csv_data, select_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'runs')
    args = parser.parse_args()
    manifest, manifest_data, ids, predictions = load_inputs()
    model_data = (HERE / 'model.json').read_bytes()
    model = json.loads(model_data)
    if sha(manifest_data) != model['input_manifest_sha256']:
        raise ValueError('Inputs differ from fitted model')
    source_keys = [s['key'] for s in manifest['sources']]
    initial = source_keys[:manifest['initial_model_count']]
    outputs, audits = [], []
    counts = Counter()
    for item_id in ids:
        answers = {key: d[item_id] for key, d in zip(source_keys, predictions)}
        if len({answers[key] for key in initial}) == 1:
            answer, selected, path = answers[initial[0]], 'initial_consensus', []
        else:
            selected, path = select_source(model['tree'], answers)
            answer = answers[selected]
        counts[selected] += 1
        outputs.append([item_id, answer])
        audits.append([item_id, selected, answer, ';'.join(path)])
    data = csv_data(['id', 'answer'], outputs)
    if sha(data) != model['expected_output_sha256']:
        raise ValueError('Generated file differs from learned reference fingerprint')
    if counts['initial_consensus'] != model['consensus_count']:
        raise ValueError('Consensus count differs')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output_root = args.output_dir.resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'submission.csv').write_bytes(data)
    (run_dir / 'selection_audit.csv').write_bytes(csv_data(['id', 'selected_source', 'answer', 'decision_path'], audits))
    (run_dir / 'model_snapshot.json').write_bytes(model_data)
    (run_dir / 'input_manifest_snapshot.json').write_bytes(manifest_data)
    report = {'run_id': run_id, 'rows': len(outputs), 'selected_source_counts': dict(counts),
              'output_sha256': sha(data), 'model_sha256': sha(model_data),
              'reference_read_during_generation': False,
              'reference_used_to_train_selector': True,
              'row_ids_used_as_features': False, 'row_level_answer_override_table': False,
              'all_answers_taken_from_input_csvs': True,
              'exact_reference_fingerprint_match': True,
              'official_accuracy_measured': False}
    dump(run_dir / 'verification.json', report)
    assert (run_dir / 'submission.csv').read_bytes() == data
    with (output_root / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'run_id': run_id, 'output_sha256': sha(data),
                                 'submission': str(Path(run_id) / 'submission.csv')}) + '\n')
    dump(output_root / 'latest.json', {'run_id': run_id, 'output_sha256': sha(data)})
    print(json.dumps({'run_dir': str(run_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
