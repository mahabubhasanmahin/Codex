"""Dataset and preprocessing shared by training, calibration and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from crossskinnet.data.metadata import apply_random_masking, get_ablation_mask, to_model_inputs
from crossskinnet.io import read_csv


class SkinDataset(Dataset):
    """Read portable dataset rows; only the training split can use metadata dropout."""

    def __init__(self, root: Path, classes: list[str], split: str, cfg: dict[str, Any],
                 mode: str = 'full', training: bool = False) -> None:
        self.root, self.classes, self.cfg = root, classes, cfg
        self.split, self.mode, self.training = split, mode, training
        if training and split != 'train':
            raise ValueError('Training-time dropout/augmentation is only allowed for the train split')
        rows = read_csv(root / ('zero_shot.csv' if split == 'zero_shot' else 'metadata.csv'))
        selected = [r for r in rows if r['split'] == split]
        if split != 'zero_shot' and any(r['source_dataset'] in ('HAM10000', 'PH2') for r in selected):
            raise ValueError('External data leaked into development CSV')
        self.excluded = sum(r['unified_label'] not in classes for r in selected)
        self.rows = [r for r in selected if r['unified_label'] in classes]
        if split != 'zero_shot' and self.excluded:
            raise ValueError(f'{split}: labels missing from training vocabulary')
        self.rng = np.random.default_rng(int(cfg['seed']))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load an image, normalized metadata and observed field mask."""
        row = self.rows[index]
        path = Path(row['image_path'])
        with Image.open(path if path.is_absolute() else self.root / path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            size = int(self.cfg['training'].get('image_size', 224))
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            if self.training and self.rng.random() < .5:
                image = ImageOps.mirror(image)
            array = np.array(image, dtype=np.float32) / 255
        metadata, mask = to_model_inputs(row, float(self.cfg.get('metadata', {}).get('age_scale', 100)))
        if self.training:
            options = self.cfg['training']
            metadata, mask = apply_random_masking(metadata, mask, options.get('p_field', .2), options.get('p_all', .2), self.rng)
        else:
            # Stable per-sample masks independent of batch ordering and backbone.
            metadata, mask = get_ablation_mask(self.mode, metadata, mask, np.random.default_rng(int(self.cfg['seed']) + index))
        mean = np.array(self.cfg['training'].get('image_mean', [.485, .456, .406]), dtype=np.float32)
        std = np.array(self.cfg['training'].get('image_std', [.229, .224, .225]), dtype=np.float32)
        tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1).copy())
        return {'image': tensor, 'metadata': torch.from_numpy(metadata), 'mask': torch.from_numpy(mask),
                'label': self.classes.index(row['unified_label']), 'index': index}
