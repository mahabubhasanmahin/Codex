"""Validate the offline fixture's leakage cases and missing metadata."""
import csv

import yaml
from PIL import Image

from tests.fixtures.make_synthetic_data import DATASETS, make_synthetic_data


def test_synthetic_data(tmp_path):
    """All five pools contain usable images, groups, masks and known duplicates."""
    root = make_synthetic_data(tmp_path / "synthetic")
    config = yaml.safe_load((root / "config.yaml").read_text())
    for name in DATASETS:
        folder = root / name
        with (folder / "metadata.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 30
        assert len({row["group_id"] for row in rows}) == 15
        assert rows[0]["age"] == "NaN" and rows[0]["age_mask"] == "0"
        assert config["datasets"][name]["pool"] == (
            "external_test" if name in ("HAM10000", "PH2") else "train")
        images = folder / "images"
        assert len(list(images.glob("*.png"))) == 30
        with Image.open(images / "image_000.png") as image:
            assert image.size == (16, 16) and image.mode == "RGB"
        assert (images / "image_029.png").read_bytes() == (images / "image_001.png").read_bytes()
    assert (root / "ISIC2018/images/image_000.png").read_bytes() == (
        root / "HAM10000/images/image_000.png").read_bytes()
