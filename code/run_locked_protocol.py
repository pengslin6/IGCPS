"""Refit the recorded DA-TGT/adapter configurations without a new search.

The configuration files are exported from retained development locks. They do
not make the repeatedly inspected temporal test tails independently unseen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('IGCPS', 'TE-CUP-SEC'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=('main', 'baselines', 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads((HERE / 'protocols' / (args.dataset + '.json')).read_text(encoding='utf-8'))
    root = args.root.resolve()
    data = root / config['fusion_csv']
    if not args.dry_run and sha256(data).lower() != config['fusion_csv_sha256'].lower():
        raise ValueError('Dataset hash does not match the recorded fused input')
    with tempfile.TemporaryDirectory(prefix='datgt-protocol-') as directory:
        temporary = Path(directory)
        lock = temporary / 'main_lock.json'
        lock.write_text(json.dumps(config['main_lock'], indent=2), encoding='utf-8')
        (temporary / 'temporal_lock.json').write_text(
            json.dumps(config['baseline_lock'], indent=2), encoding='utf-8')
        with (temporary / 'baseline_validation.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=('model', 'best_epoch'))
            writer.writeheader(); writer.writerows(config['baseline_epochs'])
        common = ['--csv', str(data), '--seeds', *map(str, config['seeds'])]
        for name, value in config['caps_per_class'].items():
            common += [f'--{name}-cap-per-class', str(value)]
        commands = []
        if args.task in ('main', 'all'):
            command = [sys.executable, '-B', '-u', str(HERE / 'hgan_causal_multiseed.py'),
                       *common, '--lock', str(lock), '--outdir', str(root / config['main_output'])]
            if args.dataset == 'TE-CUP-SEC':
                command += ['--no-calibration']
            commands.append(command)
        if args.task in ('baselines', 'all'):
            commands.append([sys.executable, '-B', '-u', str(HERE / 'paired_baseline_multiseed.py'),
                             *common, '--protocol-dir', str(temporary),
                             '--models', *[r['model'] for r in config['baseline_epochs']],
                             '--outdir', str(root / config['baseline_output'])])
        for command in commands:
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                output = Path(command[command.index('--outdir') + 1])
                if output.exists():
                    raise FileExistsError(f'Refusing to overwrite retained runs: {output}')
                subprocess.run(command, cwd=HERE, check=True)


if __name__ == '__main__':
    main()
