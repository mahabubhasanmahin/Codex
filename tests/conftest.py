"""Offline prepared pipeline fixture shared by training and evaluation tests."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import yaml

from tests.test_merge import REPO
from tests.fixtures.make_synthetic_data import make_synthetic_data
from crossskinnet.data.merge import merge_datasets, read_yaml
from crossskinnet.data.labels import inventory, apply as label_apply
from crossskinnet.data.metadata import apply as metadata_apply
from crossskinnet.data.dedup import run as dedup
from crossskinnet.data.build_unified import run as build
from crossskinnet.io import read_csv, write_csv


@pytest.fixture(scope='session')
def prepared_pipeline(tmp_path_factory) -> Path:
    """Prepare the current PAD/HAM development and PH2-only external protocol."""
    root = make_synthetic_data(tmp_path_factory.mktemp('pipeline') / 'data')
    path = root / 'config.yaml'
    cfg = read_yaml(path)
    defaults = read_yaml(REPO / 'configs/default.yaml')
    cfg.update({key: defaults[key] for key in ('dedup', 'metadata', 'model', 'training', 'evaluation', 'study')})
    for name, settings in cfg['datasets'].items():
        settings.update(pool=defaults['datasets'][name]['pool'], enabled=defaults['datasets'][name].get('enabled',True))
    # Explicit fake schemas mirror PAD patient grouping, HAM lesion grouping,
    # and PH2's image-only clinical fields without claiming real-data layouts.
    schemas_path = root / 'dataset_schemas.yaml'
    schemas = read_yaml(schemas_path)
    schemas['datasets']['PAD-UFES-20']['columns']['patient_id'] = 'group_id'
    schemas['datasets']['PH2']['columns'].pop('age')
    schemas['datasets']['PH2']['columns'].pop('lesion_id')
    schemas_path.write_text(yaml.safe_dump(schemas))
    import shutil
    shutil.copyfile(root / 'HAM10000/images/image_000.png', root / 'PH2/images/image_000.png')
    cfg['study']['target_labels'] = ['synthetic_class_0','synthetic_class_1','synthetic_class_2']
    cfg['training'].update(epochs=1, image_size=32, batch_size=16, device='cpu')
    cfg['evaluation'].update(bootstrap_repetitions=20, explanations_per_outcome=1, shap_samples=32)
    path.write_text(yaml.safe_dump(cfg))
    merge_datasets(path, root / 'dataset_schemas.yaml')
    mapping = root / 'label_mapping.csv'
    inventory(path, mapping)
    rows = read_csv(mapping)
    for row in rows:
        row.update(unified_label=row['original_label'], coarse_group='other', notes='Synthetic test-only mapping')
    write_csv(mapping, rows)
    label_apply(path, mapping)
    sites = root / 'sites.yaml'
    sites.write_text('datasets: {}\n')
    metadata_apply(path, root / 'dataset_schemas.yaml', sites)
    dedup(path)
    build(path, mapping)
    return path


@pytest.fixture(scope='session')
def trained_pipeline(prepared_pipeline: Path) -> tuple[Path, Path]:
    """Run the actual tiny training CLI once, sharing its trusted local checkpoint."""
    import subprocess
    import sys
    from crossskinnet.io import config

    args = [sys.executable, str(REPO / 'scripts/06_train.py'), '--config', str(prepared_pipeline),
            '--backbone', 'tiny', '--epochs', '1']
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checkpoint = config(prepared_pipeline)['output'] / 'runs/tiny_finetuned/best_calibrated.pt'
    return prepared_pipeline, checkpoint
