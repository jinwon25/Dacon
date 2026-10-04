"""외부 의존성 없이 공개 파일과 도달 가능한 모든 Git 이력을 검사합니다."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY = set('.csv .tsv .npz .npy .parquet .feather .pkl .pickle .joblib .pt .pth .ckpt .cbm .onnx .h5 .bin .zip .tar .gz .7z .whl .xlsx .xls .pem .key'.split())
IDENTIFIERS = {'row_id', 'pitcher_id', 'batter_id', 'player_id', 'pitcher_trackman_id', 'batter_trackman_id', 'trackman_game_id', 'game_id', 'control_success'}
PATTERNS = {
    'credential': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'private_share': re.compile(rb'https?://(?:drive\.google\.com/(?:drive/(?:u/\d+/)?folders/|file/d/)|claude\.ai/code/session_)[^\s<>"\)]+'),
}

def findings(path: str, data: bytes) -> list[str]:
    p = Path(path.lower())
    issues = []
    if (p.name.startswith('.env') and p.name != '.env.example') or p.name in {'fallback_xgb.json', 'data_description.md', 'v148_cree_xy2.py'}:
        issues.append('excluded asset or environment file')
    if p.suffix in BINARY:
        aggregate = p.suffix == '.csv' and ('/reports/' in '/' + path.lower() or path.lower().endswith('/submissions/results.csv'))
        if not aggregate:
            issues.append('data/model/prediction binary or table')
        else:
            try:
                header = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
                if IDENTIFIERS.intersection(x.strip().lower() for x in header):
                    issues.append('row identifiers or targets in CSV')
            except (UnicodeDecodeError, StopIteration, csv.Error):
                issues.append('unreadable aggregate CSV')
    if p.suffix == '.json' and b'"learner"' in data and b'"gradient_booster"' in data:
        issues.append('trained XGBoost JSON')
    for label, pattern in PATTERNS.items():
        if pattern.search(data):
            issues.append(label)
    if p.suffix == '.ipynb':
        try:
            cells = json.loads(data)['cells']
            if any(c.get('outputs') or c.get('execution_count') is not None for c in cells if c.get('cell_type') == 'code'):
                issues.append('notebook execution output')
        except (ValueError, KeyError):
            issues.append('invalid notebook')
    return issues

def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

def audit(history: bool) -> dict:
    issues = []
    paths = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode('utf-8').split('\0')
    current_count = 0
    for path in filter(None, paths):
        file = ROOT / path
        if file.is_file():
            current_count += 1
            issues.extend({'scope': 'current', 'path': path, 'reason': r} for r in findings(path, file.read_bytes()))
    objects: dict[str, set[str]] = {}
    commits = git('rev-list', '--all').decode().splitlines() if history else []
    for commit in commits:
        for entry in git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
            if not entry:
                continue
            metadata, path = entry.split(b'\t', 1)
            _, kind, oid = metadata.split()
            if kind == b'blob':
                objects.setdefault(oid.decode(), set()).add(path.decode('utf-8'))
    if objects:
        # One process for content reads, including files with several historical names.
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', '--batch'], input=('\n'.join(objects) + '\n').encode())
        position = 0
        for oid, names in objects.items():
            end = raw.index(b'\n', position)
            size = int(raw[position:end].split()[-1])
            payload = raw[end + 1:end + 1 + size]
            position = end + 2 + size
            for name in names:
                issues.extend({'scope': 'history', 'path': name, 'reason': r, 'blob': oid} for r in findings(name, payload))
        messages = git('log', '--all', '--format=%B')
        issues.extend({'scope': 'commit messages', 'path': '-', 'reason': label} for label, pattern in PATTERNS.items() if pattern.search(messages))
    return {'passed': not issues, 'current_files': current_count, 'commits': len(commits), 'historical_blobs': len(objects), 'findings': issues}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    result = audit(args.history)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
