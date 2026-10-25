"""Group-stratified development splits and a portable image dataset artifact."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from crossskinnet.io import config, present, read_csv, report, write_csv


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    """Allocate approximately 70/15/15 using stratified, disjoint group folds.

    Fewer than 20 independent groups per class reduce the ratio resolution.
    At least three independent groups per class are required; never fall back
    to an image-level split to make an undersized class appear to work.
    """
    if any(row['source_dataset'] in ('HAM10000', 'PH2') for row in rows):
        raise ValueError('External data cannot enter development splits')
    result = []
    lesion_patients: dict[str, set[str]] = {}
    for row in rows:
        if present(row.get('lesion_id')) and present(row.get('patient_id')):
            lesion_patients.setdefault(row['lesion_id'], set()).add(row['patient_id'])
    if any(len(values) > 1 for values in lesion_patients.values()):
        raise ValueError('A lesion is associated with multiple patients; review identifiers')
    for row in rows:
        group = next((row[field] for field in ('patient_id', 'lesion_id', 'image_id') if present(row.get(field))), None)
        if group is None:
            raise ValueError('Cannot determine group_id: patient_id, lesion_id and image_id are missing')
        if not present(row.get('patient_id')) and row.get('lesion_id') in lesion_patients:
            group = next(iter(lesion_patients[row['lesion_id']]))
        result.append({**row, 'group_id': group})
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


def run(config_path: Path, mapping_path: Path) -> Path:
    """Build resized development and external datasets with explicit audit artifacts."""
    cfg = config(config_path)
    output = cfg['output']
    destination = output / 'CrossSkinNet_Unified_Dataset'
    raw_train = read_csv(output / 'deduped_train_pool.csv')
    if not raw_train:
        raise ValueError('No training samples remain after deduplication')
    train = assign_splits(raw_train, int(cfg['seed']))
    external = read_csv(output / 'metadata_test_pool.csv')
    if any(r['source_dataset'] not in ('HAM10000', 'PH2') for r in external):
        raise ValueError('Non-external dataset found in zero-shot pool')
    train_labels = {r['unified_label'] for r in train if r['split'] == 'train'}
    for row in external:
        row['group_id'] = next((row[f] for f in ('patient_id', 'lesion_id', 'image_id') if present(row.get(f))), row['image_id'])
        row['split'] = 'zero_shot'
        row['zero_shot_eligible'] = str(row['unified_label'] in train_labels)
    train = _prepare_images(train, destination / 'images')
    external = _prepare_images(external, destination / 'zero_shot')
    write_csv(destination / 'metadata.csv', train)
    write_csv(destination / 'zero_shot.csv', external)
    write_csv(destination / 'label_mapping.csv', read_csv(mapping_path))
    counts = Counter((r['split'], r['source_dataset'], r['unified_label']) for r in train + external)
    stats = {'seed': cfg['seed'], 'requested_ratios': {'train': .7, 'val': .15, 'test': .15},
             'actual_split_counts': dict(Counter(r['split'] for r in train)),
             'note': 'Ratios approximate because patient/lesion groups are indivisible.',
             'zero_shot_excluded_samples': sum(r['zero_shot_eligible'] == 'False' for r in external),
             'counts': [dict(split=s, source_dataset=d, unified_label=l, count=n) for (s, d, l), n in sorted(counts.items())]}
    (destination / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
    (destination / 'README.md').write_text('# CrossSkinNet Unified Dataset\n\n'
        'Development sources: PAD-UFES-20, ISIC2018, Dermnet. HAM10000/PH2 are external only.\n'
        'See metadata.csv and zero_shot.csv. Paths are relative to this folder.\n'
        '256x256 RGB JPEGs, quality 95. NaN metadata is not statistically imputed.\n'
        'Group-level splits approximate 70/15/15; see dataset_stats.json.\n'
        'Duplicate components touching external data are excluded from development.\n'
        'Source dataset licenses and consent restrictions still apply.\n')
    _plots(train, destination)
    report('Unified development', train)
    report('Unified zero-shot', external)
    return destination
