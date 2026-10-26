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
    assert training == {'PAD-UFES-20','HAM10000'}
    assert external == {'PH2'}
    with pytest.raises(ValueError,match='disabled'):
        validate_source_rows([{'source_dataset':'ISIC2018'}],cfg,'train')
    with pytest.raises(ValueError,match='Required training sources absent'):
        validate_training_coverage([{'source_dataset':'PAD-UFES-20'}], cfg)
    cfg['datasets']['HAM10000']['pool'] = 'external_test'
    with pytest.raises(ValueError,match='Required training sources'):
        enabled_sources(cfg)
    cfg['datasets']['HAM10000']['pool'] = 'train'
    cfg['datasets']['PH2']['enabled'] = False
    with pytest.raises(ValueError,match='cannot be disabled'):
        enabled_sources(cfg)


def test_excluded_isic_not_read_and_unused_mapping_not_required(tmp_path: Path):
    """A nonexistent ISIC root is safe; only actual pool labels require mappings."""
    root = make_synthetic_data(tmp_path / 'data')
    path = root / 'config.yaml'
    cfg = yaml.safe_load(path.read_text())
    defaults = yaml.safe_load((REPO / 'configs/default.yaml').read_text())
    cfg['study'] = defaults['study']
    for name, settings in cfg['datasets'].items():
        settings.update(pool=defaults['datasets'][name]['pool'], enabled=defaults['datasets'][name].get('enabled',True))
    cfg['datasets']['Dermnet']['path'] = str(root / 'dermnet_not_attached')
    cfg['datasets']['ISIC2018'].update(enabled=False,path=str(root / 'not_attached'),exclusion_reason='User-approved overlapping source exclusion')
    path.write_text(yaml.safe_dump(cfg))
    merge_datasets(path,root / 'dataset_schemas.yaml')
    output = root / 'outputs'
    train, external = read_csv(output / 'raw_train_pool.csv'), read_csv(output / 'raw_test_pool.csv')
    assert len(train) == 60 and len(external) == 30
    assert {r['source_dataset'] for r in train} == {'PAD-UFES-20','HAM10000'}
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


def test_current_protocol_grouping_external_priority_and_fallback(prepared_pipeline: Path):
    """HAM develops; PH2 never trains; same HAM lesion cannot cross splits."""
    from crossskinnet.io import config
    from crossskinnet.train.dataset import SkinDataset
    cfg = config(prepared_pipeline)
    root = cfg['output'] / 'CrossSkinNet_Unified_Dataset'
    rows, external = read_csv(root / 'metadata.csv'), read_csv(root / 'zero_shot.csv')
    assert {r['source_dataset'] for r in rows} == {'PAD-UFES-20','HAM10000'}
    assert {r['source_dataset'] for r in external} == {'PH2'}
    assert 'HAM10000_000' not in {r['image_id'] for r in rows}
    assert len(external) == 30
    for lesion in {r['lesion_id'] for r in rows if r['source_dataset']=='HAM10000'}:
        assert len({r['split'] for r in rows if r['source_dataset']=='HAM10000' and r['lesion_id']==lesion}) == 1
    classes = sorted({r['unified_label'] for r in rows if r['split']=='train'})
    dataset = SkinDataset(root,classes,'zero_shot',cfg)
    assert all(not dataset[i]['mask'].any() for i in range(len(dataset)))
    from crossskinnet.data.grouping import assign_groups
    schemas = yaml.safe_load((REPO / 'configs/dataset_schemas.yaml').read_text())['datasets']
    with pytest.raises(ValueError,match='required grouping field lesion_id'):
        assign_groups([dict(source_dataset='HAM10000',image_id='missing',lesion_id='NaN',patient_id='NaN')],schemas)
