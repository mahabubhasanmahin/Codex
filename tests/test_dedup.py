"""Duplicate priority, metadata preference and cache repeatability."""
from pathlib import Path

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.data.merge import merge_datasets
from crossskinnet.data.dedup import BKTree, run
from crossskinnet.io import read_csv, write_csv


def test_external_duplicates_always_remove_train(tmp_path: Path):
    """ISIC/HAM overlap removes the training image and never changes external data."""
    root = make_synthetic_data(tmp_path / 'data')
    merge_datasets(root / 'config.yaml', root / 'dataset_schemas.yaml')
    output = root / 'outputs'
    for pool in ('train', 'test'):
        rows = read_csv(output / f'raw_{pool}_pool.csv')
        # Internal image 29 duplicates 1; make the later record more complete.
        for row in rows:
            if row['image_id'].endswith('_001'):
                row['age'] = 'NaN'
            if row['image_id'].endswith('_029'):
                row['age'] = '40'
        write_csv(output / f'metadata_{pool}_pool.csv', rows)
    external_before = (output / 'metadata_test_pool.csv').read_bytes()
    run(root / 'config.yaml')
    rows = read_csv(output / 'deduped_train_pool.csv')
    ids = {r['image_id'] for r in rows}
    assert 'ISIC2018_000' not in ids
    assert 'PAD-UFES-20_029' in ids and 'PAD-UFES-20_001' not in ids
    assert (output / 'metadata_test_pool.csv').read_bytes() == external_before
    assert not any(r['source_dataset'] in ('HAM10000', 'PH2') for r in rows)
    assert any(r['action'] == 'remove_train_external_overlap' for r in read_csv(output / 'duplicates_report.csv'))
    before = (output / 'image_hashes.csv').read_bytes()
    run(root / 'config.yaml')
    assert (output / 'image_hashes.csv').read_bytes() == before


def test_tree_matches_reference():
    """BK-tree neighbors match an exhaustive reference on a small hash set."""
    values = [0, 1, 3, 7, 255, 2**63, 2**64 - 1]
    tree = BKTree()
    for i, value in enumerate(values):
        tree.add(value, i)
    for value in values:
        assert sorted(tree.query(value, 2)) == sorted((i, (value ^ other).bit_count())
            for i, other in enumerate(values) if (value ^ other).bit_count() <= 2)


def test_excluded_external_labels_still_protect_training(tmp_path: Path):
    """Label exclusion cannot hide external images from the overlap reference pool."""
    root = make_synthetic_data(tmp_path / 'data')
    merge_datasets(root / 'config.yaml', root / 'dataset_schemas.yaml')
    output = root / 'outputs'
    write_csv(output / 'metadata_train_pool.csv', read_csv(output / 'raw_train_pool.csv'))
    # Simulate all HAM diagnoses being excluded at harmonization.
    write_csv(output / 'metadata_test_pool.csv', [r for r in read_csv(output / 'raw_test_pool.csv') if r['source_dataset'] == 'PH2'])
    run(root / 'config.yaml')
    assert 'ISIC2018_000' not in {r['image_id'] for r in read_csv(output / 'deduped_train_pool.csv')}
