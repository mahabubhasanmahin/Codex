"""Exercise real ingestion code against explicitly configured synthetic data."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.fixtures.make_synthetic_data import make_synthetic_data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from crossskinnet.data.merge import merge_datasets


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a generated pool preserving missing-value literals."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_cli_pools_and_metadata(tmp_path: Path) -> None:
    """The CLI keeps external sources isolated and preserves labels and missingness."""
    root = make_synthetic_data(tmp_path / "data")
    schemas_path = root / "dataset_schemas.yaml"
    schemas = yaml.safe_load(schemas_path.read_text())
    # Add a symptom only PAD provides and a second original partition to ISIC.
    pad = root / "PAD-UFES-20/metadata.csv"
    rows = read_rows(pad)
    for row in rows:
        row["itch"] = "NaN" if row["age_mask"] == "0" else "true"
    with pad.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    schemas["datasets"]["PAD-UFES-20"]["symptoms"] = {"itch": "itch"}
    source = root / "ISIC2018/metadata.csv"
    (source.parent / "original_test.csv").write_bytes(source.read_bytes())
    schemas["datasets"]["ISIC2018"]["metadata_files"].append("original_test.csv")
    schemas_path.write_text(yaml.safe_dump(schemas))
    result = subprocess.run([sys.executable, str(REPO / "scripts/01_merge.py"),
        "--config", str(root / "config.yaml"), "--schemas", str(schemas_path)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    train = read_rows(root / "outputs/raw_train_pool.csv")
    external = read_rows(root / "outputs/raw_test_pool.csv")
    assert len(train) == 120 and len(external) == 60
    assert {r["source_dataset"] for r in train} == {"PAD-UFES-20", "ISIC2018", "Dermnet"}
    assert {r["source_dataset"] for r in external} == {"HAM10000", "PH2"}
    assert train[0]["age"] == "NaN" and train[0]["age_mask"] == "0"
    assert train[1]["itch"] == "true" and train[1]["itch_mask"] == "1"
    assert external[0]["itch"] == "NaN" and external[0]["itch_mask"] == "0"
    assert train[0]["patient_id"] == "NaN" and train[0]["sex"] == "NaN"
    assert train[0]["lesion_id"] == "PAD-UFES-20_group_000"
    assert train[0]["original_label"] == "synthetic_class_0"
    assert all(Path(r["image_path"]).is_file() for r in train + external)
    assert "Counts per source_dataset" in result.stdout
    assert "Counts per original_label" in result.stdout


def test_missing_real_schema_fails(tmp_path: Path) -> None:
    """Absent real inspection information cannot silently select synthetic layouts."""
    root = make_synthetic_data(tmp_path / "data")
    with pytest.raises(ValueError, match="Supply real inspection output"):
        missing = root / "missing_schemas.yaml"
        missing.write_text(yaml.safe_dump({"datasets": {}}))
        merge_datasets(root / "config.yaml", missing)
    assert not (root / "outputs").exists()


def test_missing_column_and_wrong_pool_fail(tmp_path: Path) -> None:
    """Reject incorrect column mappings and attempts to train on external sources."""
    root = make_synthetic_data(tmp_path / "data")
    config_path = root / "config.yaml"
    schemas_path = root / "dataset_schemas.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["datasets"]["PH2"]["pool"] = "train"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="pool must be external_test"):
        merge_datasets(config_path, schemas_path)
    config["datasets"]["PH2"]["pool"] = "external_test"
    config_path.write_text(yaml.safe_dump(config))
    schemas = yaml.safe_load(schemas_path.read_text())
    schemas["datasets"]["PAD-UFES-20"]["columns"]["age"] = "nonexistent"
    schemas_path.write_text(yaml.safe_dump(schemas))
    with pytest.raises(ValueError, match="missing configured columns"):
        merge_datasets(config_path, schemas_path)
    assert not (root / "outputs").exists()
