"""Offline prepared pipeline fixture shared by training and evaluation tests."""
from pathlib import Path

import pytest
import yaml

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.data.merge import merge_datasets, read_yaml
from crossskinnet.data.labels import inventory, apply as label_apply
from crossskinnet.data.metadata import apply as metadata_apply
from crossskinnet.data.dedup import run as dedup
from crossskinnet.data.build_unified import run as build
from crossskinnet.io import read_csv, write_csv


@pytest.fixture(scope='session')
def prepared_pipeline(tmp_path_factory) -> Path:
    """Prepare all five synthetic sources with test-only, non-medical labels."""
    root = make_synthetic_data(tmp_path_factory.mktemp('pipeline') / 'data')
    path = root / 'config.yaml'
    cfg = read_yaml(path)
    defaults = read_yaml(REPO / 'configs/default.yaml')
    cfg.update({key: defaults[key] for key in ('dedup', 'metadata', 'model', 'training')})
    cfg['training'].update(epochs=1, image_size=32, batch_size=16, device='cpu')
    path.write_text(yaml.safe_dump(cfg))
    merge_datasets(path, root / 'dataset_schemas.yaml')
    mapping = root / 'label_mapping.csv'
    inventory(path, mapping)
    rows = read_csv(mapping)
    for row in rows:
        row.update(unified_label=row['original_label'], coarse_group='other', notes='Synthetic test-only mapping')
    write_csv(mapping, rows)
    label_apply(path, mapping)
    sites = root / 'sites.yaml'
    sites.write_text('datasets: {}\n')
    metadata_apply(path, root / 'dataset_schemas.yaml', sites)
    dedup(path)
    build(path, mapping)
    return path
