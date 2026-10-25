"""Disjoint group splitting and portable image export."""
from pathlib import Path

import pytest
from PIL import Image
import yaml

from tests.test_merge import REPO
from crossskinnet.data.build_unified import assign_splits, run
from crossskinnet.io import read_csv, write_csv


def test_group_splits_and_resize(tmp_path: Path):
    """Multiple images per patient stay together; external labels cannot train."""
    output = tmp_path / 'out'
    rows = []
    for label in ('a', 'b'):
        for group in range(20):
            path = tmp_path / f'{label}_{group}.png'
            Image.new('RGB', (12, 20), color=(group * 10, 10, 50)).save(path)
            for image in range(2):
                rows.append(dict(image_id=f'{label}_{group}_{image}', image_path=str(path), source_dataset='PAD-UFES-20',
                    unified_label=label, coarse_group='other', patient_id=f'{label}_p{group}', lesion_id='NaN'))
    output.mkdir()
    write_csv(output / 'deduped_train_pool.csv', rows)
    write_csv(output / 'metadata_test_pool.csv', [{**rows[0], 'source_dataset':'HAM10000', 'unified_label':'unseen'}])
    mapping = tmp_path / 'mapping.csv'
    write_csv(mapping, [{'unified_label':'a'}])
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'seed':42, 'paths':{'output_root':str(output)}}))
    destination = run(cfg, mapping)
    dataset = read_csv(destination / 'metadata.csv')
    assert {r['split'] for r in dataset} == {'train', 'val', 'test'}
    groups = {r['group_id']: r['split'] for r in dataset}
    assert all(groups[r['group_id']] == r['split'] for r in dataset)
    assert assign_splits(rows, 42) == assign_splits(rows, 42)
    with Image.open(destination / dataset[0]['image_path']) as image:
        assert image.size == (256, 256)
    assert read_csv(destination / 'zero_shot.csv')[0]['zero_shot_eligible'] == 'False'
    with pytest.raises(ValueError, match='External'):
        assign_splits([{**rows[0], 'source_dataset':'PH2'}],42)
    with pytest.raises(ValueError, match='at least 3'):
        assign_splits(rows[:2],42)
