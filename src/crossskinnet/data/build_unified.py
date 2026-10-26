"""Group-stratified development splits and a portable image dataset artifact."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from crossskinnet.io import config, present, read_csv, report, write_csv
from crossskinnet.data.grouping import assign_groups
from crossskinnet.data.sources import validate_target_labels, validate_source_rows, validate_training_coverage, enabled_sources
from crossskinnet.data.merge import read_yaml


def assign_splits(rows: list[dict[str, str]], seed: int, schemas: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Allocate approximately 70/15/15 using stratified, disjoint group folds.

    Fewer than 20 independent groups per class reduce the ratio resolution.
    At least three independent groups per class are required; never fall back
    to an image-level split to make an undersized class appear to work.
    """
    if any(row['source_dataset'] == 'PH2' for row in rows):
        raise ValueError('External data cannot enter development splits')
    result = assign_groups(rows, schemas)
    labels = np.array([r['unified_label'] for r in result])
    groups = np.array([r['group_id'] for r in result])
    counts = {label: len(set(groups[labels == label])) for label in set(labels)}
    folds = min(20, min(counts.values(), default=0))
    if folds < 3:
        raise ValueError(f'Need at least 3 independent groups per class for train/val/test; found {counts}')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    val_folds = max(1, round(folds * .15))
    test_folds = max(1, round(folds * .15))
    training_folds = folds - val_folds - test_folds
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        split = 'train' if fold < training_folds else ('val' if fold < training_folds + val_folds else 'test')
        for index in indices:
            result[index]['split'] = split
    group_splits: dict[str, set[str]] = {}
    for row in result:
        group_splits.setdefault(row['group_id'], set()).add(row['split'])
    assert all(len(splits) == 1 for splits in group_splits.values()), 'Group leakage detected'
    if set(labels) != {r['unified_label'] for r in result if r['split'] == 'train'}:
        raise ValueError('A label is absent from training after group stratification; collect more groups or revise taxonomy')
    return result


def _prepare_images(rows: list[dict[str, Any]], folder: Path) -> list[dict[str, Any]]:
    """Resize the shorter side, center crop 256 pixels, and write unique JPEG names."""
    folder.mkdir(parents=True, exist_ok=True)
    result = []
    for row in rows:
        identity = f"{row['source_dataset']}:{row['image_id']}:{row['image_path']}"
        filename = hashlib.sha256(identity.encode()).hexdigest() + '.jpg'
        with Image.open(row['image_path']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            scale = 256 / min(image.size)
            image = image.resize(tuple(max(256, round(size * scale)) for size in image.size), Image.Resampling.LANCZOS)
            image = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
            image.save(folder / filename, quality=95)
        result.append({**row, 'original_image_path': row['image_path'], 'image_path': f'{folder.name}/{filename}'})
    return result


def _plots(rows: list[dict[str, Any]], output: Path) -> None:
    """Save class counts per split and per source without interactive displays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for category in ('split', 'source_dataset'):
        labels = sorted({r['unified_label'] for r in rows})
        keys = sorted({r[category] for r in rows})
        figure, axis = plt.subplots(figsize=(max(8, len(labels) * .5), 5))
        bottom = np.zeros(len(labels))
        for key in keys:
            counts = Counter(r['unified_label'] for r in rows if r[category] == key)
            heights = np.array([counts[label] for label in labels])
            axis.bar(labels, heights, bottom=bottom, label=key)
            bottom += heights
        axis.tick_params(axis='x', rotation=90)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f'class_distribution_{category}.png')
        plt.close(figure)


def run(config_path: Path, mapping_path: Path, schemas_path: Path | None = None) -> Path:
    """Build resized development and external datasets with explicit audit artifacts."""
    cfg = config(config_path)
    if cfg.get('review_preview_only'):
        raise ValueError('Review preview is unnormalized; run stages 03 and 04 with the main config before building')
    output = cfg['output']
    destination = output / 'CrossSkinNet_Unified_Dataset'
    status_file = output / 'dedup_status.json'
    if not status_file.exists():
        raise ValueError('Missing dedup_status.json; rerun stage 04 with the review-aware policy')
    status = json.loads(status_file.read_text())
    if status.get('policy') not in ('review','quarantine') or not status.get('ready'):
        raise ValueError('Duplicate review is incomplete; review candidates and rerun stage 04 before building')
    if status.get('reviews_csv'):
        review_file = Path(status['reviews_csv'])
        if not review_file.exists() or hashlib.sha256(review_file.read_bytes()).hexdigest() != status.get('reviews_sha256'):
            raise ValueError('Duplicate review decisions changed; rerun stage 04 before building')
    train_file = output / 'deduped_train_pool.csv'
    if hashlib.sha256(train_file.read_bytes()).hexdigest() != status.get('train_pool_sha256'):
        raise ValueError('Deduped training CSV differs from the verified status; rerun stage 04')
    raw_train = read_csv(train_file)
    validate_source_rows(raw_train, cfg, 'train')
    if not raw_train:
        raise ValueError('No training samples remain after deduplication')
    schema_file = schemas_path or config_path.resolve().parent / 'dataset_schemas.yaml'
    if schemas_path is not None or schema_file.exists():
        schemas = read_yaml(schema_file)['datasets']
    else:
        schemas = {}
    if status['policy']=='quarantine':
        if not schema_file.is_file() or hashlib.sha256(schema_file.read_bytes()).hexdigest()!=status.get('grouping_schema_sha256'):
            raise ValueError('Grouping schema differs from group quarantine; rerun stage 04 with the intended schema')
    validate_target_labels(raw_train, cfg, 'Post-dedup development')
    train = assign_splits(raw_train, int(cfg['seed']), schemas)
    validate_target_labels([row for row in train if row['split']=='train'], cfg, 'Training split')
    validate_training_coverage([row for row in train if row['split']=='train'], cfg)
    external = read_csv(output / 'metadata_test_pool.csv')
    validate_source_rows(external, cfg, 'test')
    actual_sources = sorted({r['source_dataset'] for r in train})
    actual_external = sorted({r['source_dataset'] for r in external})
    train_labels = {r['unified_label'] for r in train if r['split'] == 'train'}
    for row in external:
        row['group_id'] = next((row[f] for f in ('patient_id', 'lesion_id', 'image_id') if present(row.get(f))), row['image_id'])
        row['split'] = 'zero_shot'
        row['zero_shot_eligible'] = str(row['unified_label'] in train_labels)
    train = _prepare_images(train, destination / 'images')
    external = _prepare_images(external, destination / 'zero_shot')
    (destination / 'grouping_schema.yaml').write_text(yaml.safe_dump({
        'datasets': {source: schema.get('grouping', {}) for source, schema in schemas.items()}}))
    write_csv(destination / 'metadata.csv', train)
    write_csv(destination / 'zero_shot.csv', external)
    integrity = dict(policy=status['policy'], ready=True, files={name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                     for name in ('metadata.csv','zero_shot.csv')})
    (destination / 'dataset_integrity.json').write_text(json.dumps(integrity, indent=2))
    write_csv(destination / 'label_mapping.csv', read_csv(mapping_path))
    counts = Counter((r['split'], r['source_dataset'], r['unified_label']) for r in train + external)
    stats = {'seed': cfg['seed'], 'dedup_policy':status['policy'],
             'quarantined_training_rows':status.get('quarantined_training_rows',0), 'actual_development_sources': actual_sources, 'actual_external_sources': actual_external, 'requested_ratios': {'train': .7, 'val': .15, 'test': .15},
             'actual_split_counts': dict(Counter(r['split'] for r in train)),
             'note': 'Ratios approximate because patient/lesion groups are indivisible.',
             'zero_shot_excluded_samples': sum(r['zero_shot_eligible'] == 'False' for r in external),
             'counts': [dict(split=s, source_dataset=d, unified_label=l, count=n) for (s, d, l), n in sorted(counts.items())]}
    (destination / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
    (destination / 'README.md').write_text('# CrossSkinNet Unified Dataset\n\n'
        f"Actual development sources: {', '.join(actual_sources)}. External only: {', '.join(actual_external)}.\n"
        'See metadata.csv and zero_shot.csv. Paths are relative to this folder.\n'
        '256x256 RGB JPEGs, quality 95. NaN metadata is not statistically imputed.\n'
        'Group-level splits approximate 70/15/15; see dataset_stats.json.\n'
        'Verified duplicate components touching external data are excluded from development.\n'
        f"Cohort policy: {status['policy']}; group-quarantine exclusions: {status.get('quarantined_training_rows',0)}.\n"
        'Group quarantine exclusions are not duplicate verdicts. Report cohort selection limits.\n'
        'Source dataset licenses and consent restrictions still apply.\n')
    _plots(train, destination)
    report('Unified development', train)
    report('Unified zero-shot', external)
    return destination
