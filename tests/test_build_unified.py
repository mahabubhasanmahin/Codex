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
    import hashlib, json
    (output / 'dedup_status.json').write_text(json.dumps(dict(policy='review',ready=True,
        train_pool_sha256=hashlib.sha256((output / 'deduped_train_pool.csv').read_bytes()).hexdigest())))
    write_csv(output / 'metadata_test_pool.csv', [{**rows[0], 'source_dataset':'PH2', 'unified_label':'unseen'}])
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


def test_patient_scoped_lesions_do_not_merge_unrelated_patients():
    """Repeated local lesion numbers are valid only with explicit source scope."""
    from crossskinnet.data.grouping import assign_groups
    rows = [dict(image_id=f'PAT_{patient}_1_{image}.png', source_dataset='PAD-UFES-20',
                 patient_id=f'PAT_{patient}', lesion_id='1', unified_label=label)
            for patient in range(20) for image, label in enumerate(('a','b'))]
    schemas = {'PAD-UFES-20': {'grouping': {'lesion_id_scope':'patient',
                'image_id_prefix_template':'{patient_id}_{lesion_id}_'}}}
    with pytest.raises(ValueError, match='global lesion ID'):
        assign_splits(rows,42)
    split = assign_splits(rows,42,schemas)
    assert len({r['group_id'] for r in split}) == 20
    for patient in range(20):
        assert len({r['split'] for r in split if r['patient_id'] == f'PAT_{patient}'}) == 1
    with pytest.raises(ValueError, match='requires patient_id'):
        assign_groups([{**rows[0], 'patient_id':'NaN'}], schemas)
    with pytest.raises(ValueError, match='prefix'):
        assign_groups([{**rows[0], 'image_id':'wrong.png'}], schemas)
