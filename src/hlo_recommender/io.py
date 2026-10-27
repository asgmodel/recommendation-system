"""Configuration, data and run provenance."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import random
import tomllib
from datetime import datetime, timezone
from pathlib import Path

def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON at {path}:{line_number}') from exc
    return rows

def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n')

def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join((json.dumps(row, ensure_ascii=True) + '\n' for row in rows)))

def config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open('rb') as stream:
        result = tomllib.load(stream)
    result['root'] = str(path.parent.parent)
    return result

def resolve(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(cfg['root']) / path

def new_run(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Run directory must be empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def provenance(*, seed: int, files: dict[str, str | Path], **fields) -> dict:
    versions = {}
    for package in ('hlo-recommender', 'rdflib', 'numpy', 'scipy', 'torch', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'python': platform.python_version(), 'versions': versions, 'input_sha256': {key: digest(value) for key, value in files.items()}, **fields}

def grouped_split(rows: list[dict], group_key: str, seed: int, validation_fraction: float=0.2, test_fraction: float=0.2) -> dict:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError('Split fractions must be between zero and one')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('Training fraction must be positive')
    groups = sorted({row[group_key] for row in rows})
    if len(groups) < 5:
        raise ValueError('At least five independent groups are required')
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * validation_fraction))
    n_test = max(1, round(len(groups) * test_fraction))
    if n_val + n_test >= len(groups):
        raise ValueError('Insufficient training groups')
    partition = {g: 'validation' for g in groups[:n_val]}
    partition.update({g: 'test' for g in groups[n_val:n_val + n_test]})
    partition.update({g: 'train' for g in groups[n_val + n_test:]})
    return {name: [r for r in rows if partition[r[group_key]] == name] for name in ('train', 'validation', 'test')}
