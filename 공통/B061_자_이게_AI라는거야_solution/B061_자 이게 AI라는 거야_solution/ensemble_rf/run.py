"""Build RF_v1 from its saved plan, or rerun RF training and score optimization."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / '01_rf_v1_original_run'
SNAPSHOT = HERE / '02_input_snapshots'
RUNTIME = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resolve_records(value):
    if isinstance(value, list):
        return [resolve_records(v) for v in value]
    if isinstance(value, dict):
        result = {k: resolve_records(v) for k, v in value.items()}
        if 'path' in result and 'sha256' in result:
            path = (HERE / result['path']).resolve()
            if not path.is_relative_to(HERE):
                raise ValueError(f'Input must be in the package: {path}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != result['sha256']:
                raise ValueError(f'Input changed: {path}')
            result['path'] = str(path)
        return result
    return value


def node_environment(args):
    node = Path(args.node).expanduser().resolve() if args.node else RUNTIME / 'bin/node'
    modules = Path(args.node_modules).expanduser().resolve() if args.node_modules else RUNTIME / 'node_modules'
    if not node.is_file() or not (modules / '@oai/artifact-tool').is_dir():
        raise RuntimeError('Node.js and @oai/artifact-tool are required. Set --node and --node-modules.')
    return node, modules


def prepare_training(run, manifest):
    """Connect the unchanged training algorithm to the frozen local inputs."""
    solution = run / 'context/solution'
    sources = [dict(key=r['key'], name=r['name'], source_path=r['path'], sha256=r['sha256'])
               for r in manifest['consensus_models']]
    write_json(solution / 'ensemble_merge/reference_matching_experiment/input_manifest.json',
               {'sources': sources})
    scores = run / 'context/scores'
    records = [{k: v for k, v in r.items() if k not in {'conditional_correct', 'verified_local_sha256'}}
               for r in manifest['score_records']]
    write_json(scores / 'score_records.json', records)
    write_json(scores / 'probe_feedback.json', {'scores': {}})
    write_json(scores / 'probes.json', {'queries': []})
    write_json(scores / 'active_probe_config.json', {'experiment_manifest': str(scores / 'probes.json')})
    validation = SNAPSHOT / 'validation_data'
    config = read_json(validation / 'inputs.json')
    for key in ('validation', 'test_questions'):
        path = validation / config[key]['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key]['sha256']:
            raise ValueError(f'Validation input changed: {path}')
    research = (ARCHIVE / 'research.py').read_text(encoding='utf-8')
    old = "R=next(Path('/Users/yunjongseo/Downloads').glob('B061_*solution'))\nS=W/'score_audit_20260927'; V=R/'ensemble_merge/type_based_ensemble'"
    new = f"R=P/'context/solution'\nS=P/'context/scores'; V=Path({str(validation)!r})"
    if research.count(old) != 1:
        raise ValueError('Unexpected research.py layout')
    research = research.replace(old, new)
    old_agnes = "Path('/Users/yunjongseo/Downloads/submission_agnes3_highres1024_tta4.csv')"
    if research.count(old_agnes) != 1:
        raise ValueError('Unexpected auxiliary input definition')
    research = research.replace(old_agnes, f"Path({manifest['auxiliary'][0]['path']!r})")
    (run / 'research.py').write_text(research, encoding='utf-8')
    shutil.copy2(ARCHIVE / 'robust.py', run / 'robust.py')
    certify = (ARCHIVE / 'certify.py').read_text(encoding='utf-8')
    certify = certify.replace('submission_public_score_plus2.csv', 'RF_v1.csv')
    certify = certify.replace('submission_public_score_12changes.csv', 'RF_v1.csv')
    (run / 'certify.py').write_text(certify, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Rerun RF training, score projection and optimization using the original 60 scored submissions.')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare an isolated run without executing it.')
    parser.add_argument('--check', action='store_true', help='Validate inputs and runtime without creating a run.')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'runs', help='Parent directory for separate timestamped runs.')
    parser.add_argument('--node', help='Node.js executable override.')
    parser.add_argument('--node-modules', help='Directory containing @oai/artifact-tool.')
    args = parser.parse_args()
    manifest = resolve_records(read_json(SNAPSHOT / 'relative_input_manifest.json'))
    plan = resolve_records(read_json(SNAPSHOT / 'relative_delivery_plan.json'))
    if len(manifest['unique_score_records']) != 60:
        raise ValueError('Expected the pre-RF_v1 snapshot with 60 scored submissions')
    node, modules = node_environment(args)
    if args.retrain:
        missing = [m for m in ('numpy', 'scipy', 'sklearn', 'joblib') if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError('Install training dependencies: python3 -m pip install -r ensemble_rf/requirements.txt')
    if args.check:
        print('Inputs and runtime verified. Training snapshot: 60 scored submissions.')
        return
    run_id = datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    run = args.output_dir.expanduser().resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / 'node_modules').symlink_to(modules, target_is_directory=True)
    shutil.copy2(ARCHIVE / 'build.mjs', run / 'build.mjs')
    write_json(run / 'input_manifest.json', manifest)
    plan['output_file'] = 'RF_v1.csv'
    # A new local run has no newly observed public score.
    for key in ('observed_public_score', 'score_source', 'observed_public_score_source'):
        plan.pop(key, None)
    write_json(run / 'delivery_plan.json', plan)
    if args.retrain:
        prepare_training(run, manifest)
    status = {'mode': 'retrain' if args.retrain else 'saved_plan', 'run_directory': str(run),
              'historical_scored_submissions': 60, 'status': 'prepared', 'submitted': False}
    write_json(run / 'run_status.json', status)
    print(f'Run directory: {run}', flush=True)
    if args.prepare_only:
        return
    try:
        if args.retrain:
            for script in ('research.py', 'robust.py', 'certify.py'):
                print(f'Running {script}', flush=True)
                subprocess.run([sys.executable, str(run / script)], cwd=run, check=True)
        subprocess.run([str(node), str(run / 'build.mjs')], cwd=run, check=True)
        result = run / 'RF_v1.csv'
        status['sha256'] = hashlib.sha256(result.read_bytes()).hexdigest()
        status['matches_archived_RF_v1'] = result.read_bytes() == (ARCHIVE / 'RF_v1.csv').read_bytes()
        if not args.retrain and not status['matches_archived_RF_v1']:
            raise ValueError('Saved-plan output differs from archived RF_v1')
        status['status'] = 'completed'
        status['output'] = str(result)
    except Exception as exc:
        status['status'] = 'failed'
        status['error'] = str(exc)
        raise
    finally:
        write_json(run / 'run_status.json', status)
    print(f'Output: {result}', flush=True)


if __name__ == '__main__':
    main()
