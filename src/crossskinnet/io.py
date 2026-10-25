"""Configuration, CSV persistence and reproducible reporting helpers."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def config(path: Path) -> dict[str, Any]:
    """Resolve configured runtime output paths relative to the YAML file."""
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if 'seed' not in cfg:
        raise ValueError('Missing seed in config')
    output = Path(cfg['paths']['output_root'])
    cfg['output'] = output if output.is_absolute() else path.resolve().parent / output
    cfg['output'].mkdir(parents=True, exist_ok=True)
    return cfg


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while preserving NaN literals and empty strings."""
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write rows with an explicit union of fields, also for empty results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        raise ValueError(f'Cannot write {path}: no columns specified')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(name: str, rows: list[dict[str, Any]]) -> None:
    """Print shape, representative rows and categorical counts."""
    print(f'{name}: shape=({len(rows)}, {len(rows[0]) if rows else 0}); head={rows[:2]}')
    for field in ('source_dataset', 'original_label', 'unified_label', 'split'):
        print(field, dict(Counter(str(row.get(field, '')) for row in rows)))


def present(value: Any) -> bool:
    """Recognize missing serialized metadata without imputing values."""
    return value is not None and str(value).strip().lower() not in ('', 'nan', 'na', 'none', 'null', 'unknown', 'unk')
