"""Actual audit CLI on synthetic sources, without medical mapping decisions."""
from pathlib import Path
import subprocess
import sys

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.io import read_csv


def test_audit_cli_without_mappings(tmp_path: Path):
    """Report post-overlap counts without changing reviewed intermediates or config."""
    root = make_synthetic_data(tmp_path / 'data')
    output = root / 'outputs'
    output.mkdir()
    sentinel = output / 'metadata_train_pool.csv'
    sentinel.write_text('existing normalized data must stay unchanged\n')
    before = (root / 'config.yaml').read_bytes()
    args = [sys.executable, str(REPO / 'scripts/00_audit.py'), '--config', str(root / 'config.yaml'),
            '--schemas', str(root / 'dataset_schemas.yaml')]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / 'config.yaml').read_bytes() == before
    assert sentinel.read_text() == 'existing normalized data must stay unchanged\n'
    audit = output / 'audit'
    summaries = {r['source_dataset']: r for r in read_csv(audit / 'source_summary.csv')}
    assert summaries['HAM10000']['raw_images'] == summaries['HAM10000']['retained_images'] == '30'
    assert summaries['ISIC2018']['retained_images'] == '28'
    counts = read_csv(audit / 'class_feasibility.csv')
    isic = next(r for r in counts if r['source_dataset'] == 'ISIC2018' and r['original_label'] == 'synthetic_class_0')
    assert isic['removed_external_overlap'] == '1'
    assert sum(int(r['raw_images']) for r in counts) == 90
    metadata = read_csv(audit / 'metadata_availability.csv')
    pad_age = next(r for r in metadata if r['source_dataset'] == 'PAD-UFES-20' and r['field'] == 'age')
    assert pad_age['observed'] == '22' and pad_age['missing'] == '8'
    assert all(r['disease_label_approved'] == 'False' for r in read_csv(audit / 'label_review.csv'))
    from zipfile import ZipFile
    with ZipFile(audit / 'audit_reports.zip') as archive:
        assert len(archive.namelist()) == 6
        assert 'REPORT.md' in archive.namelist()
        assert 'metadata_train_pool.csv' not in archive.namelist()
    assert (audit / 'REPORT.md').exists()
    assert (audit / 'duplicate_pairs.png').exists()
    from crossskinnet.data.audit import review_overlap
    review_overlap(root / 'config.yaml')
    assert (audit / 'overlap_evidence.csv').exists()
    assert (audit / 'PAD-UFES-20_overlap_review.png').exists()
    first = (audit / 'class_feasibility.csv').read_bytes()
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (audit / 'class_feasibility.csv').read_bytes() == first


def test_overlap_evidence_distinguishes_exact_and_transitive_edges():
    """A perceptual chain is not reported as a proven exact external duplicate."""
    from crossskinnet.data.audit import overlap_evidence
    raw = [dict(source_dataset='ISIC2018',image_id='a',image_path='a.png'),
           dict(source_dataset='PAD-UFES-20',image_id='b',image_path='b.png')]
    pairs = [dict(source_a='ISIC2018',image_id_a='a',image_path_a='a.png',source_b='HAM10000',
                  image_id_b='c',image_path_b='c.png',distance='0',exact_sha256='True'),
             dict(source_a='PAD-UFES-20',image_id_a='b',image_path_a='b.png',source_b='ISIC2018',
                  image_id_b='a',image_path_b='a.png',distance='3',exact_sha256='False')]
    result = {(r['source_dataset'],r['evidence']): r['training_images_connected_to_external'] for r in overlap_evidence(raw,pairs)}
    assert result[('ISIC2018','reported_sha256_edges')] == 1
    assert result[('PAD-UFES-20','reported_sha256_edges')] == 0
    assert result[('PAD-UFES-20','all_reported_phash_edges')] == 1
