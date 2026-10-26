"""Shared-three-class mapping and interactive per-pair review with tiny fake images."""
from pathlib import Path
import random

from PIL import Image
import pytest
import yaml

from tests.test_merge import REPO
from crossskinnet.data.labels import apply
from crossskinnet.data.review import prepare, ReviewSession
from crossskinnet.data.dedup import _reviewed, _pair_id
from crossskinnet.data.build_unified import run as build
from crossskinnet.io import read_csv, write_csv


@pytest.fixture
def clinical_label_fixture(tmp_path: Path) -> Path:
    """Use real source code strings on fake pixels, never actual medical data."""
    cfg = yaml.safe_load((REPO / 'configs/default.yaml').read_text())
    output = tmp_path / 'outputs'
    output.mkdir()
    cfg['paths']['output_root'] = str(output)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    labels = {'PAD-UFES-20':['MEL','BCC','NEV','ACK','SCC','SEK'],
              'HAM10000':['mel','bcc','nv','akiec','bkl','df','vasc'],
              'PH2':['Melanoma','Common Nevus','Atypical Nevus']}
    rows = []
    rng = random.Random(cfg['seed'])
    for source, values in labels.items():
        for label in values:
            path = tmp_path / f'{source}_{label}.png'
            image = Image.frombytes('RGB',(16,16),rng.randbytes(16*16*3))
            if label=='BCC': image=Image.new('RGB',(16,16),'red')
            if label=='bcc': image=Image.new('RGB',(16,16),'blue')
            image.save(path)
            rows.append(dict(source_dataset=source,original_label=label,image_id=path.stem,
                image_path=str(path),patient_id=f'{source}_{label}' if source=='PAD-UFES-20' else 'NaN',
                lesion_id=f'{source}_{label}' if source!='PH2' else 'NaN',age='40',sex='male',site='NaN'))
    # The excluded PH2 atypical label must still protect against training overlap.
    (tmp_path / 'PH2_Atypical Nevus.png').write_bytes((tmp_path / 'HAM10000_nv.png').read_bytes())
    write_csv(output / 'raw_train_pool.csv',[r for r in rows if r['source_dataset']!='PH2'])
    write_csv(output / 'raw_test_pool.csv',[r for r in rows if r['source_dataset']=='PH2'])
    return config


def test_canonical_mapping_counts_and_exclusions(clinical_label_fixture: Path):
    """Only the declared three disease labels are included, with explicit exclusions."""
    cfg = clinical_label_fixture
    output = cfg.parent / 'outputs'
    apply(cfg,REPO / 'configs/label_mapping.csv')
    training = read_csv(output / 'labeled_train_pool.csv')
    assert len(training)==6 and len(read_csv(output / 'labeled_test_pool.csv'))==2
    assert {r['unified_label'] for r in training}=={'melanoma','basal_cell_carcinoma','melanocytic_nevus'}
    excluded = read_csv(output / 'label_exclusions.csv')
    assert sum(int(r['count']) for r in excluded if r['pool']=='train')==7
    assert next(r for r in excluded if r['original_label']=='Atypical Nevus')['count']=='1'
    assert len(read_csv(output / 'raw_test_pool.csv'))==3


def test_prepare_selected_scope_and_save_decisions(clinical_label_fixture: Path, monkeypatch):
    """Filter candidates, preserve raw external references and never auto-decide pHash."""
    cfg = clinical_label_fixture
    output = cfg.parent / 'outputs'
    from crossskinnet.data.dedup import cached_hashes
    cache = output / 'audit/image_hashes.csv'
    cache.parent.mkdir()
    cached_hashes(read_csv(output / 'raw_train_pool.csv')+read_csv(output / 'raw_test_pool.csv'), cache)
    def unexpected_hash(path):
        raise AssertionError('Fingerprint-valid audit hashes should be reused')
    monkeypatch.setattr('crossskinnet.data.dedup._hash_file',unexpected_hash)
    (output / 'metadata_train_pool.csv').write_text('existing normalized table\n')
    folder = prepare(cfg,REPO / 'configs/label_mapping.csv')
    assert (output / 'metadata_train_pool.csv').read_text()=='existing normalized table\n'
    assert len(read_csv(folder / 'raw_test_pool.csv'))==3
    assert 'HAM10000_nv' not in {r['image_id'] for r in read_csv(folder / 'provisional_train_pool.csv')}
    session = ReviewSession(cfg)
    assert len(session.rows)==1 and not session.decisions().get(session.rows[0]['pair_id'])
    review_file = output / 'near_duplicate_reviews.csv'
    records = read_csv(review_file)
    historical = {**records[0], 'sha256_a':'a'*64,'sha256_b':'b'*64,
                  'pair_id':_pair_id('a'*64,'b'*64),'is_duplicate':'True','notes':'Unrelated prior review'}
    write_csv(review_file,records+[historical])
    pair = session.rows[0]['pair_id']
    session.save(pair,False,'Different generated solid colors')
    decisions, saved = _reviewed(review_file)
    assert decisions[pair]=='false' and decisions[historical['pair_id']]=='true'
    assert saved[pair]['notes']=='Different generated solid colors'
    with pytest.raises(ValueError,match='not in'):
        session.save('invalid',True,'')
    with pytest.raises(ValueError,match='Decision must'):
        session.save(pair,1,'')
    with pytest.raises(ValueError,match='preview is unnormalized'):
        build(folder / 'preview_config.yaml',REPO / 'configs/label_mapping.csv')
    session.save(pair,None,'Uncertain; leave pending')
    assert session.decisions()[pair]==''
    current = yaml.safe_load(cfg.read_text())
    current['seed']+=1
    cfg.write_text(yaml.safe_dump(current))
    with pytest.raises(ValueError,match='Review input changed'):
        ReviewSession(cfg)


def test_widget_click_is_an_explicit_saved_verdict(clinical_label_fixture: Path):
    """The notebook panel renders and its Different button saves only that pair."""
    from crossskinnet.data.review_ui import launch_review
    prepare(clinical_label_fixture,REPO / 'configs/label_mapping.csv')
    session = ReviewSession(clinical_label_fixture)
    panel = launch_review(clinical_label_fixture)
    panel.children[3].value='Synthetic color collision; different images'
    panel.children[4].children[1].click()
    assert session.decisions()[session.rows[0]['pair_id']]=='false'
    panel.close()
