"""Offline miniature datasets that exercise the inspected production schemas."""
import csv
from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook
from PIL import Image

from tests.test_merge import REPO, read_rows
from crossskinnet.data.merge import merge_datasets, load_isic2018, load_ham10000


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write configured metadata into a tiny fake Kaggle tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image(path: Path) -> None:
    """Create an offline RGB image, including BMP where required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), 'red').save(path)


@pytest.fixture
def inspected_tree(tmp_path: Path) -> tuple[Path, dict]:
    """Mirror all observed ingestion formats without real medical images."""
    schemas = yaml.safe_load((REPO / 'configs/dataset_schemas.yaml').read_text())['datasets']
    for name, schema in schemas.items():
        root = tmp_path / name
        root.mkdir()
        if name == 'PAD-UFES-20':
            row = {column: 'x' for column in schema['columns'].values()}
            row.update(img_id='PAT_1_1.png', patient_id='PAT_1', lesion_id='1', diagnostic='NEV', age='NaN')
            row.update({column: 'UNK' for column in schema['symptoms'].values()})
            write_csv(root / schema['metadata_files'][0], [row])
            image(root / schema['image_template'].format(value=row['img_id']))
        elif name == 'ISIC2018':
            row = {'image': 'ISIC_1', **{label: '0.0' for label in schema['one_hot_labels']}}
            row['NV'] = '1.0'
            write_csv(root / schema['metadata_files'][0], [row])
            write_csv(root / schema['join']['file'], [{'image': 'ISIC_1', 'lesion_id': 'HAM_1'}])
            image(root / schema['image_template'].format(value='ISIC_1'))
        elif name == 'HAM10000':
            write_csv(root / schema['metadata_files'][0], [{'image_id':'ISIC_1', 'lesion_id':'HAM_1',
                'dx':'nv', 'age':'40', 'sex':'unknown', 'localization':'back'}])
            image(root / 'HAM10000_images_part_1/ISIC_1.jpg')
            # Mirrored lower-case directory must not double the external pool.
            image(root / 'ham10000_images_part_1/ISIC_1.jpg')
        elif name == 'Dermnet':
            for split in ('train', 'test'):
                image(root / split / 'Acne and Rosacea Photos/example.jpg')
        else:
            image(root / 'PH2Dataset/PH2 Dataset images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp')
            image(root / 'PH2Dataset/PH2 Dataset images/IMD002/IMD002_lesion/IMD002_lesion.bmp')
            book = Workbook()
            sheet = book.active
            sheet.title = schema['sheet']
            sheet.append(['Title'])
            sheet.append(['Image Name', 'Histological Diagnosis', 'Clinical Diagnosis'])
            sheet.append([None, None, 'Common Nevus', 'Atypical Nevus', 'Melanoma'])
            sheet.append(['IMD002', 'Melanoma', None, None, 'X'])
            book.save(root / schema['metadata_file'])
    config = {'paths': {'output_root': str(tmp_path / 'outputs')}, 'datasets': {
        name: {'path': str(tmp_path / name), 'pool': 'external_test' if name in ('HAM10000', 'PH2') else 'train'}
        for name in schemas}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, schemas


def test_inspected_merge(inspected_tree: tuple[Path, dict]) -> None:
    """Production schemas join IDs, decode labels, merge splits and exclude masks."""
    config, _ = inspected_tree
    train_path, test_path = merge_datasets(config, REPO / 'configs/dataset_schemas.yaml')
    train, external = read_rows(train_path), read_rows(test_path)
    assert len(train) == 4 and len(external) == 2
    assert train[0]['itch'] == 'NaN' and train[0]['itch_mask'] == '0'
    assert train[1]['original_label'] == 'NV' and train[1]['lesion_id'] == 'HAM_1'
    assert train[2]['original_label'] == train[3]['original_label'] == 'Acne and Rosacea Photos'
    assert train[2]['image_id'] != train[3]['image_id']
    assert external[0]['original_label'] == 'nv' and external[0]['sex_mask'] == '0'
    assert external[1]['original_label'] == 'Melanoma'
    assert '_Dermoscopic_Image' in external[1]['image_path']


def test_ambiguous_images_and_labels(inspected_tree: tuple[Path, dict]) -> None:
    """Reject silent image selection and invalid one-hot diagnosis rows."""
    config, schemas = inspected_tree
    root = config.parent
    image(root / 'HAM10000/HAM10000_images_part_2/ISIC_1.jpg')
    with pytest.raises(ValueError, match='Ambiguous image ID'):
        load_ham10000(root / 'HAM10000', schemas['HAM10000'])
    schema = schemas['ISIC2018']
    row = {'image':'ISIC_1', **{label:'1' for label in schema['one_hot_labels']}}
    write_csv(root / 'ISIC2018' / schema['metadata_files'][0], [row])
    with pytest.raises(ValueError, match='exactly one active label'):
        load_isic2018(root / 'ISIC2018', schema)
