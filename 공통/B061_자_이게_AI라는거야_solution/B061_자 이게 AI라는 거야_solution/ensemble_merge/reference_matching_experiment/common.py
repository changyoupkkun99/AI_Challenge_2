import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_submission(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != ['id', 'answer']:
        raise ValueError('Invalid submission columns')
    rows = list(reader)
    ids = [r['id'] for r in rows]
    answers = {r['id']: r['answer'] for r in rows}
    if len(rows) != 6714 or len(answers) != 6714 or set(ids) != {f'test_{i:04d}.jpg' for i in range(1, 6715)}:
        raise ValueError('Invalid IDs')
    if not set(answers.values()) <= set('abcd'):
        raise ValueError('Invalid answers')
    return ids, answers


def load_inputs():
    manifest_data = (HERE / 'input_manifest.json').read_bytes()
    manifest = json.loads(manifest_data)
    all_predictions = []
    ids = None
    for source in manifest['sources']:
        relative = Path(source['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid input path')
        data = (HERE / relative).read_bytes()
        if sha(data) != source['sha256']:
            raise ValueError('Input changed: ' + source['key'])
        row_ids, answers = read_submission(data)
        if ids is None:
            ids = row_ids
        elif ids != row_ids:
            raise ValueError('Input order differs')
        all_predictions.append(answers)
    return manifest, manifest_data, ids, all_predictions


def csv_data(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def select_source(tree, answers):
    path = []
    while 'selected_source_key' not in tree:
        yes = answers[tree['feature_source_key']] == tree['equals_answer']
        path.append(f"{tree['feature_source_key']}={tree['equals_answer']}:{int(yes)}")
        tree = tree['yes'] if yes else tree['no']
    return tree['selected_source_key'], path
