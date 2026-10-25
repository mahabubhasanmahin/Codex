"""Mapping completeness, explicit exclusions and zero-shot class eligibility."""
from pathlib import Path

import pytest

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.data.merge import merge_datasets
from crossskinnet.data.labels import apply, inventory
from crossskinnet.io import read_csv, write_csv


def test_label_review_and_exclusions(tmp_path: Path) -> None:
    """Inventory preserves edits; apply rejects unknowns and reports all exclusions."""
    root = make_synthetic_data(tmp_path / 'data')
    cfg, mapping = root / 'config.yaml', root / 'mapping.csv'
    merge_datasets(cfg, root / 'dataset_schemas.yaml')
    inventory(cfg, mapping)
    with pytest.raises(ValueError, match='fill unified_label'):
        apply(cfg, mapping)
    rows = read_csv(mapping)
    for row in rows:
        row.update(unified_label=row['original_label'], coarse_group='other', notes='Synthetic test labels only')
        if row['source_dataset'] == 'PH2':
            row['unified_label'] = 'external_only'
        if row['source_dataset'] == 'Dermnet' and row['original_label'] == 'synthetic_class_0':
            row['include'] = 'False'
    write_csv(mapping, rows)
    inventory(cfg, mapping)
    assert read_csv(mapping) == rows
    apply(cfg, mapping)
    output = root / 'outputs'
    assert len(read_csv(output / 'labeled_train_pool.csv')) == 80
    assert len(read_csv(output / 'labeled_test_pool.csv')) == 60
    assert sum(r['zero_shot_eligible'] == 'False' for r in read_csv(output / 'labeled_test_pool.csv')) == 30
    assert read_csv(output / 'label_exclusions.csv')[0]['count'] == '10'
    assert any(r['reason'] == 'no_training_counterpart_exclude_zero_shot' for r in read_csv(output / 'label_warnings.csv'))
    write_csv(mapping, rows[:-1])
    with pytest.raises(ValueError, match='Missing label mappings'):
        apply(cfg, mapping)
