"""Explicit, editable label harmonization; no medical equivalences are inferred."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from crossskinnet.io import config, read_csv, report, write_csv

FIELDS = ['source_dataset', 'original_label', 'unified_label', 'coarse_group', 'include', 'notes']
GROUPS = {'malignant', 'benign', 'inflammatory', 'infectious', 'other'}


def inventory(config_path: Path, mapping_path: Path) -> None:
    """List raw labels and add unmapped entries without overwriting edits."""
    output = config(config_path)['output']
    rows = read_csv(output / 'raw_train_pool.csv') + read_csv(output / 'raw_test_pool.csv')
    counts = Counter((row['source_dataset'], row['original_label']) for row in rows)
    existing = read_csv(mapping_path) if mapping_path.exists() else []
    known = {(r['source_dataset'], r['original_label']) for r in existing}
    for (source, label), count in sorted(counts.items()):
        print(f'{source}: {label!r}: {count}')
        if (source, label) not in known:
            existing.append(dict(zip(FIELDS, [source, label, '', '', 'True',
                'TODO: fill unified_label and coarse_group or set include=False with a reason.'])))
    write_csv(mapping_path, existing, FIELDS)
    write_csv(output / 'original_label_counts.csv', [dict(source_dataset=s, original_label=l, count=n)
              for (s, l), n in sorted(counts.items())], ['source_dataset', 'original_label', 'count'])
    report('Label inventory', existing)


def _mapping(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Validate every reviewed mapping and reject duplicate or incomplete entries."""
    mapping = {}
    for row in read_csv(path):
        if not set(FIELDS) <= row.keys():
            raise ValueError(f'{path}: expected columns {FIELDS}')
        key = (row['source_dataset'], row['original_label'])
        if key in mapping:
            raise ValueError(f'Duplicate mapping: {key}')
        if row['include'].lower() not in ('true', 'false'):
            raise ValueError(f'{key}: include must be True or False')
        if row['include'].lower() == 'true' and (not row['unified_label'].strip() or row['coarse_group'] not in GROUPS):
            raise ValueError(f'{key}: fill unified_label and coarse_group in {path}, or explicitly exclude')
        mapping[key] = row
    groups: dict[str, str] = {}
    for row in mapping.values():
        if row['include'].lower() == 'true':
            label, group = row['unified_label'], row['coarse_group']
            if label in groups and groups[label] != group:
                raise ValueError(f'Conflicting coarse groups for unified label {label}')
            groups[label] = group
    return mapping


def apply(config_path: Path, mapping_path: Path) -> None:
    """Map both isolated pools and report every exclusion and unsupported test class."""
    output = config(config_path)['output']
    mapping = _mapping(mapping_path)
    pools = [read_csv(output / f'raw_{pool}_pool.csv') for pool in ('train', 'test')]
    missing = sorted({(r['source_dataset'], r['original_label']) for rows in pools for r in rows} - mapping.keys())
    if missing:
        raise ValueError(f'Missing label mappings: {missing}; run inventory and review {mapping_path}')
    mapped: list[list[dict[str, Any]]] = []
    exclusions = []
    for pool, rows in zip(('train', 'test'), pools):
        kept = []
        removed: Counter = Counter()
        for row in rows:
            entry = mapping[(row['source_dataset'], row['original_label'])]
            if entry['include'].lower() == 'false':
                removed[(row['source_dataset'], row['original_label'])] += 1
                continue
            kept.append({**row, 'unified_label': entry['unified_label'], 'coarse_group': entry['coarse_group']})
        for (source, label), count in removed.items():
            print(f'include=False removed {count} {pool} rows: {source} / {label}')
            exclusions.append(dict(pool=pool, source_dataset=source, original_label=label, count=count))
        mapped.append(kept)
    train_labels = {r['unified_label'] for r in mapped[0]}
    for row in mapped[1]:
        row['zero_shot_eligible'] = str(row['unified_label'] in train_labels)
    for pool, rows, raw in zip(('train', 'test'), mapped, pools):
        fields = [*raw[0], 'unified_label', 'coarse_group'] if raw else ['unified_label', 'coarse_group']
        if pool == 'test':
            fields.append('zero_shot_eligible')
        write_csv(output / f'labeled_{pool}_pool.csv', rows, fields)
        report(f'Labeled {pool}', rows)
    counts = Counter((r['unified_label'], r['source_dataset']) for rows in mapped for r in rows)
    write_csv(output / 'label_counts.csv', [dict(unified_label=l, source_dataset=s, count=n)
              for (l, s), n in sorted(counts.items())], ['unified_label', 'source_dataset', 'count'])
    totals = Counter(r['unified_label'] for r in mapped[0])
    warnings = [dict(unified_label=l, reason='fewer_than_50_training_samples', count=n)
                for l, n in sorted(totals.items()) if n < 50]
    unseen = Counter(r['unified_label'] for r in mapped[1] if r['unified_label'] not in train_labels)
    warnings += [dict(unified_label=l, reason='no_training_counterpart_exclude_zero_shot', count=n)
                 for l, n in sorted(unseen.items())]
    write_csv(output / 'label_warnings.csv', warnings, ['unified_label', 'reason', 'count'])
    write_csv(output / 'label_exclusions.csv', exclusions, ['pool', 'source_dataset', 'original_label', 'count'])
    print('Label warnings:', warnings)
