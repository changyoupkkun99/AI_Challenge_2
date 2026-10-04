#!/usr/bin/env python3
"""Verify an exact impossibility certificate without an optimizer."""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata


def resolve(root, relative):
    for part in Path(relative).parts:
        found = [p for p in root.iterdir()
                 if unicodedata.normalize('NFC', p.name) == unicodedata.normalize('NFC', part)]
        if len(found) != 1:
            raise ValueError('Missing or ambiguous path: ' + relative)
        root = found[0]
    return root


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['id', 'answer']
        rows = list(reader)
    answers = {r['id']: r['answer'] for r in rows}
    assert len(rows) == len(answers) == 6714
    assert set(answers) == {f'test_{i:04d}.jpg' for i in range(1, 6715)}
    assert set(answers.values()) <= set('abcd')
    return answers


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parent.parent
    certificate = json.loads((folder / 'certificate.json').read_text(encoding='utf-8'))
    config_path = folder.parent / 'merge_config.json'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == certificate['config_sha256']
    sources = json.loads(config_path.read_text(encoding='utf-8'))['sources']
    reference_path = resolve(root, certificate['reference_path'])
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == certificate['reference_sha256']
    reference = read(reference_path)
    data = []
    for source in sources:
        path = resolve(root, source['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        data.append(read(path))
    combined = [0] * len(sources)
    for term in certificate['terms']:
        item_id, target_index, rival_index = term['example']
        target, rival = chr(target_index + 97), chr(rival_index + 97)
        assert reference[item_id] == target and target != rival
        multiplier = term['multiplier']
        assert type(multiplier) is int and multiplier > 0
        actual = [int(p[item_id] == target) - int(p[item_id] == rival) for p in data]
        assert actual == term['coefficients']
        combined = [v + multiplier * c for v, c in zip(combined, actual)]
    assert combined == certificate['combined_coefficients']
    assert all(c < 0 for c in combined)
    print(json.dumps({
        'exact_integer_certificate_verified': True,
        'sources_verified': len(sources),
        'conflicting_rows': [t['example'][0] for t in certificate['terms']],
        'nonnegative_nonzero_global_weights_feasible': False,
        'holds_even_if_every_tie_favors_reference': True,
        'input_or_merge_files_changed': False,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
