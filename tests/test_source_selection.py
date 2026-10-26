"""Revised study selection, no access to excluded ISIC and stale-pool rejection."""
from pathlib import Path

import pytest
import yaml

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.data.merge import merge_datasets
from crossskinnet.data.labels import inventory, apply
from crossskinnet.data.sources import enabled_sources, validate_source_rows, validate_training_coverage
from crossskinnet.io import read_csv, write_csv


def test_default_two_source_policy():
    """The user-approved default no longer requires current ISIC2018 as input."""
    cfg = yaml.safe_load((REPO / 'configs/default.yaml').read_text())
    training, external = enabled_sources(cfg)
    assert training == {'PAD-UFES-20','Dermnet'}
    assert external == {'HAM10000','PH2'}
    with pytest.raises(ValueError,match='disabled'):
        validate_source_rows([{'source_dataset':'ISIC2018'}],cfg,'train')
    with pytest.raises(ValueError,match='Required training sources absent'):
        validate_training_coverage([{'source_dataset':'PAD-UFES-20'}], cfg)
    cfg['datasets']['HAM10000']['enabled'] = False
    with pytest.raises(ValueError,match='cannot be disabled'):
        enabled_sources(cfg)


def test_excluded_isic_not_read_and_unused_mapping_not_required(tmp_path: Path):
    """A nonexistent ISIC root is safe; only actual pool labels require mappings."""
    root = make_synthetic_data(tmp_path / 'data')
    path = root / 'config.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['datasets']['ISIC2018'].update(enabled=False,path=str(root / 'not_attached'),exclusion_reason='User-approved overlapping source exclusion')
    path.write_text(yaml.safe_dump(cfg))
    merge_datasets(path,root / 'dataset_schemas.yaml')
    output = root / 'outputs'
    train, external = read_csv(output / 'raw_train_pool.csv'), read_csv(output / 'raw_test_pool.csv')
    assert len(train) == 60 and len(external) == 60
    assert {r['source_dataset'] for r in train} == {'PAD-UFES-20','Dermnet'}
    record = next(r for r in read_csv(output / 'source_selection.csv') if r['source_dataset']=='ISIC2018')
    assert record['enabled']=='False' and record['reason']
    mapping = root / 'mapping.csv'
    inventory(path,mapping)
    entries = read_csv(mapping)
    for row in entries:
        row.update(unified_label=row['original_label'],coarse_group='other',notes='Synthetic test only')
    entries.append(dict(source_dataset='ISIC2018',original_label='unused',unified_label='',
                        coarse_group='',include='True',notes='Unused historical mapping; deliberately unknown'))
    write_csv(mapping,entries)
    apply(path,mapping)
    assert len(read_csv(output / 'labeled_train_pool.csv'))==60
    # A legacy pool cannot silently reintroduce the disabled source.
    write_csv(output / 'raw_train_pool.csv',train + [{**train[0],'source_dataset':'ISIC2018'}])
    with pytest.raises(ValueError,match='rerun merge'):
        apply(path,mapping)
