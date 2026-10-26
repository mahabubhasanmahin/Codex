"""One-command synthetic pipeline demonstration, never a thesis experiment."""
from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import yaml

from crossskinnet.data.synthetic import make_synthetic_data
from crossskinnet.data.merge import merge_datasets, read_yaml
from crossskinnet.data.labels import inventory, apply as apply_labels
from crossskinnet.data.metadata import apply as apply_metadata
from crossskinnet.data.dedup import run as dedup
from crossskinnet.data.build_unified import run as build
from crossskinnet.train.engine import run as train
from crossskinnet.eval.evaluate import run as evaluate
from crossskinnet.io import config, read_csv, write_csv


def run(config_path: Path) -> Path:
    """Create fresh fake inputs below output_root and exercise stages 01–07."""
    base = config(config_path)
    directory = Path(tempfile.mkdtemp(prefix='SYNTHETIC_DEMO_', dir=base['output']))
    inputs = make_synthetic_data(directory / 'inputs', seed=int(base['seed']))
    cfg = read_yaml(inputs / 'config.yaml')
    for section in ('dedup', 'metadata', 'model', 'training', 'evaluation'):
        cfg[section] = base[section]
    if 'study' in base:
        cfg['study'] = {**base['study'], 'target_labels':['synthetic_class_0','synthetic_class_1','synthetic_class_2']}
    for name, settings in cfg['datasets'].items():
        settings['pool'] = base.get('datasets', {}).get(name, {}).get('pool', settings['pool'])
        settings['enabled'] = base.get('datasets', {}).get(name, {}).get('enabled', True)
        settings['exclusion_reason'] = base.get('datasets', {}).get(name, {}).get('exclusion_reason', '')
    cfg['paths']['output_root'] = str(directory / 'outputs')
    cfg['training'].update(epochs=1, image_size=32, batch_size=16)
    cfg['evaluation'].update(bootstrap_repetitions=20, explanations_per_outcome=1, shap_samples=32)
    path = directory / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    schemas = inputs / 'dataset_schemas.yaml'
    schema_data = read_yaml(schemas)
    schema_data['datasets']['PAD-UFES-20']['columns']['patient_id'] = 'group_id'
    schema_data['datasets']['PH2']['columns'].pop('age', None)
    schema_data['datasets']['PH2']['columns'].pop('lesion_id', None)
    schemas.write_text(yaml.safe_dump(schema_data))
    if cfg['datasets']['HAM10000']['pool'] == 'train':
        shutil.copyfile(inputs / 'HAM10000/images/image_000.png', inputs / 'PH2/images/image_000.png')
    merge_datasets(path, schemas)
    mapping = directory / 'label_mapping.csv'
    inventory(path, mapping)
    rows = read_csv(mapping)
    for row in rows:
        row.update(unified_label=row['original_label'], coarse_group='other', notes='SYNTHETIC TEST ONLY; no medical diagnosis mapping.')
    write_csv(mapping, rows)
    apply_labels(path, mapping)
    sites = directory / 'site_mapping.yaml'
    sites.write_text('datasets: {}\n')
    apply_metadata(path, schemas, sites)
    dedup(path)
    build(path, mapping, schemas)
    checkpoint = train(path, 'tiny', epochs=1)
    evaluate(path, checkpoint)
    (directory / 'README.md').write_text('# SYNTHETIC DEMO ONLY\n\n'
        'These random images and arbitrary class labels validate software execution.\n'
        'Their metrics are not scientific results and must never be used in the thesis.\n'
        'Real runs require reviewed medical labels, site mappings and source datasets.\n')
    print(f'SYNTHETIC DEMO ONLY. Complete artifacts: {directory}')
    return directory
