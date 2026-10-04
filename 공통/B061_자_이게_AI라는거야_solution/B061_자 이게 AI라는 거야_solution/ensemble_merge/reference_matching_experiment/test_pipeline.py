"""Check portability, provenance, consensus preservation and input tamper detection."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class PipelineTest(unittest.TestCase):
    def test_portable_generation_and_input_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'standalone'
            root.mkdir()
            for name in ['common.py', 'generate.py', 'input_manifest.json', 'model.json']:
                shutil.copyfile(HERE / name, root / name)
            shutil.copytree(HERE / 'inputs', root / 'inputs')
            # No reference, training program, workspace or absolute paths are copied.
            command = [sys.executable, str(root / 'generate.py')]
            runs = [json.loads(subprocess.check_output(command, cwd=temp)) for _ in range(2)]
            self.assertNotEqual(runs[0]['run_id'], runs[1]['run_id'])
            first = Path(runs[0]['run_dir'])
            result = (first / 'submission.csv').read_bytes()
            self.assertEqual(result, (Path(runs[1]['run_dir']) / 'submission.csv').read_bytes())
            model = json.loads((root / 'model.json').read_text())
            self.assertEqual(hashlib.sha256(result).hexdigest(), model['expected_output_sha256'])
            history = (root / 'runs/history.jsonl').read_text().splitlines()
            self.assertEqual(len(history), 2)
            sources = json.loads((root / 'input_manifest.json').read_text())['sources']
            inputs = {s['key']: {r['id']: r['answer'] for r in csv.DictReader(
                io.StringIO((root / s['path']).read_text(encoding='utf-8-sig')))} for s in sources}
            outputs = list(csv.DictReader(io.StringIO(result.decode())))
            audits = list(csv.DictReader(io.StringIO((first / 'selection_audit.csv').read_text())))
            self.assertEqual(len(outputs), 6714)
            self.assertEqual(len({r['id'] for r in outputs}), 6714)
            self.assertEqual([r['id'] for r in outputs], list(inputs[sources[0]['key']]))
            initial = [s['key'] for s in sources[:10]]
            locked = 0
            for row, audit in zip(outputs, audits):
                item_id = row['id']
                self.assertEqual(audit['id'], item_id)
                self.assertIn(row['answer'], 'abcd')
                unanimous = {inputs[k][item_id] for k in initial}
                if len(unanimous) == 1:
                    locked += 1
                    self.assertEqual(audit['selected_source'], 'initial_consensus')
                    self.assertEqual(row['answer'], next(iter(unanimous)))
                else:
                    node = model['tree']
                    while 'selected_source_key' not in node:
                        branch = 'yes' if inputs[node['feature_source_key']][item_id] == node['equals_answer'] else 'no'
                        node = node[branch]
                    self.assertEqual(audit['selected_source'], node['selected_source_key'])
                    self.assertEqual(row['answer'], inputs[node['selected_source_key']][item_id])
            self.assertEqual(locked, 4444)
            original = root / sources[0]['path']
            original.write_bytes(original.read_bytes() + b'\n')
            failed = subprocess.run(command, cwd=temp, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Input changed: model_01', failed.stderr)
            self.assertEqual(len((root / 'runs/history.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
