"""Generate an explicitly synthetic layout; this does not describe real datasets."""
from __future__ import annotations

import csv
import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image
import yaml

DATASETS = ("PAD-UFES-20", "ISIC2018", "Dermnet", "HAM10000", "PH2")


def make_synthetic_data(root: Path | None = None, seed: int = 42) -> Path:
    """Create 30 RGB images per dataset in a new directory and return its path.

    With no root, create a temporary directory (the caller owns cleanup).
    Each dataset has one byte-identical internal duplicate. HAM10000 image 0
    also copies ISIC2018 image 0, deliberately simulating external-test leakage.
    Groups have two images each. Missing ages remain NaN with an explicit mask.
    """
    destination = Path(tempfile.mkdtemp(prefix="crossskinnet-")) if root is None else Path(root)
    if root is not None:
        destination.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    config: dict = {"seed": seed, "paths": {"kaggle_input_root": str(destination),
                    "output_root": str(destination / "outputs")}, "datasets": {}}
    for name in DATASETS:
        folder = destination / name
        images = folder / "images"
        images.mkdir(parents=True)
        rows = []
        for index in range(30):
            filename = f"image_{index:03}.png"
            target = images / filename
            if index == 29:
                shutil.copyfile(images / "image_001.png", target)
            elif name == "HAM10000" and index == 0:
                shutil.copyfile(destination / "ISIC2018/images/image_000.png", target)
            else:
                Image.frombytes("RGB", (16, 16), rng.randbytes(16 * 16 * 3)).save(target)
            missing = index % 4 == 0
            rows.append({"image_id": f"{name}_{index:03}", "image_path": f"images/{filename}",
                         "group_id": f"{name}_group_{index // 2:03}",
                         "label": f"synthetic_class_{index % 3}",
                         "age": "NaN" if missing else str(rng.randint(18, 85)),
                         "age_mask": 0 if missing else 1})
        with (folder / "metadata.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config["datasets"][name] = {"path": str(folder),
            "pool": "external_test" if name in ("HAM10000", "PH2") else "train",
            "metadata_csv": "metadata.csv"}
        print(f"{name}: images=(30, 16, 16, 3), metadata=(30, 6), groups=15; head={rows[:2]}")
    schemas = {"datasets": {name: {"format": "csv", "metadata_files": ["metadata.csv"],
        "columns": {"image_id": "image_id", "image_path": "image_path",
                    "original_label": "label", "lesion_id": "group_id", "age": "age"}}
        for name in DATASETS}}
    (destination / "dataset_schemas.yaml").write_text(yaml.safe_dump(schemas, sort_keys=False))
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


if __name__ == "__main__":
    print(make_synthetic_data())
