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


def test_phash_collision_requires_review_and_blocks_stale_pool(tmp_path: Path):
    """Different solid images share pHash but cannot be auto-removed, even at d=0."""
    import json
    import pytest
    import yaml
    from PIL import Image
    from crossskinnet.data.build_unified import run as build
    output = tmp_path / 'output'
    output.mkdir()
    first, second = tmp_path / 'red.png', tmp_path / 'blue.png'
    Image.new('RGB',(16,16),'red').save(first)
    Image.new('RGB',(16,16),'blue').save(second)
    train = dict(image_id='a',image_path=str(first),source_dataset='PAD-UFES-20')
    external = dict(image_id='b',image_path=str(second),source_dataset='PH2')
    write_csv(output / 'metadata_train_pool.csv',[train])
    write_csv(output / 'raw_test_pool.csv',[external])
    (output / 'deduped_train_pool.csv').write_text('stale old output\n')
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'seed':42,'paths':{'output_root':str(output)}}))
    with pytest.raises(ValueError,match='require review'):
        run(cfg)
    status = json.loads((output / 'dedup_status.json').read_text())
    assert not status['ready'] and status['pending_relevant_pairs'] == 1
    assert (output / 'deduped_train_pool.csv').read_text() == 'stale old output\n'
    assert len(read_csv(output / 'provisional_train_pool.csv')) == 1
    with pytest.raises(ValueError,match='review is incomplete'):
        build(cfg,tmp_path / 'absent_mapping.csv')
    reviews = output / 'near_duplicate_reviews.csv'
    rows = read_csv(reviews)
    assert rows[0]['distance'] == '0'
    rows[0]['is_duplicate'] = 'False'
    rows[0]['notes'] = 'Different generated solid colors; intentional perceptual collision'
    write_csv(reviews,rows)
    run(cfg)
    assert json.loads((output / 'dedup_status.json').read_text())['ready']
    assert read_csv(output / 'deduped_train_pool.csv')[0]['image_id'] == 'a'
    rows[0]['is_duplicate'] = 'True'
    write_csv(reviews,rows)
    run(cfg)
    assert not read_csv(output / 'deduped_train_pool.csv')
    assert len(read_csv(output / 'raw_test_pool.csv')) == 1


def test_all_distinct_sha_candidates_in_collision_bucket():
    """Rejecting a bucket representative cannot hide another close SHA pair."""
    from crossskinnet.data.dedup import _candidate_edges
    rows = [dict(image_path=str(index)) for index in range(3)]
    hashes = {str(index): dict(sha256=str(index),phash='8000000000000000') for index in range(3)}
    exact, candidates = _candidate_edges(rows,hashes,5,2)
    assert not exact
    assert {(a,b) for a,b,_ in candidates} == {(0,1),(0,2),(1,2)}
    # Exact SHA grouping is never dependent on the pHash representative.
    hashes['1']['sha256'] = hashes['0']['sha256']
    hashes['1']['phash'] = 'ffffffffffffffff'
    exact, _ = _candidate_edges(rows,hashes,5,2)
    assert exact[0][:2] == (0,1)
