"""Explicit metadata normalization, structural missingness and field-level masking."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crossskinnet.io import config, present, read_csv, report, write_csv
from crossskinnet.data.merge import read_yaml
from crossskinnet.data.sources import validate_source_rows

SYMPTOMS = ('itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation')
FIELDS = ('age', 'sex', 'site', *SYMPTOMS)
SITES = ('head_neck', 'trunk', 'upper_limb', 'lower_limb', 'hands_feet', 'other')
SEXES = ('male', 'female', 'unknown')
WIDTHS = (1, 3, 6, 1, 1, 1, 1, 1, 1)
SLICES = tuple(slice(sum(WIDTHS[:i]), sum(WIDTHS[:i + 1])) for i in range(len(FIELDS)))
MODES = ('full', 'drop_age', 'drop_sex', 'drop_site', 'drop_symptoms', 'all_missing',
         'random_25', 'random_50', 'random_75')


def inventory(config_path: Path, mapping_path: Path) -> None:
    """Print observed raw site values and preserve existing reviewed site mappings."""
    output = config(config_path)['output']
    mapping = read_yaml(mapping_path) if mapping_path.exists() else {'datasets': {}}
    for pool in ('train', 'test'):
        rows = read_csv(output / f'labeled_{pool}_pool.csv')
        for row in rows:
            if present(row['site']):
                mapping['datasets'].setdefault(row['source_dataset'], {}).setdefault(row['site'], None)
    for source, sites in mapping['datasets'].items():
        print(source, 'Raw sites requiring a mapping:', [site for site, value in sites.items() if value is None])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text('# TODO: map every null raw site to head_neck, trunk, upper_limb, lower_limb, hands_feet or other.\n'
                            + yaml.safe_dump(mapping, sort_keys=False))


def _normalize(row: dict[str, str], schema: dict[str, Any], sites: dict[str, str]) -> dict[str, Any]:
    """Normalize one row without filling missing values with population statistics."""
    result: dict[str, Any] = dict(row)
    available = set(schema.get('columns', {})) | set(schema.get('symptoms', {}))
    for field in FIELDS:
        value = row.get(field, 'NaN')
        missing = not present(value)
        normalized: Any = 'NaN'
        if not missing:
            if field == 'age':
                normalized = float(value)
                if not np.isfinite(normalized) or normalized < 0:
                    raise ValueError(f"Invalid age {value!r} for {row['image_id']}")
            elif field == 'sex':
                normalized = value.lower()
                if normalized not in SEXES:
                    raise ValueError(f"Unknown sex {value!r} for {row['source_dataset']}; configure a normalizer")
            elif field == 'site':
                normalized = sites.get(value)
                if normalized not in SITES:
                    raise ValueError(f"Map raw site {value!r} for {row['source_dataset']} in site_mapping.yaml")
            else:
                normalized = {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}.get(value.lower())
                if normalized is None:
                    raise ValueError(f'Invalid binary symptom {field}={value!r}')
        elif field == 'sex':
            normalized = 'unknown'
        result[field] = normalized
        result[f'{field}_missing'] = int(missing)
        result[f'{field}_mask'] = int(not missing)
        result[f'structurally_missing_{field}'] = int(field not in available)
        if field not in available and not missing:
            raise ValueError(f"{row['source_dataset']}: {field} has values but is absent from dataset schema")
    return result


def apply(config_path: Path, schemas_path: Path, mapping_path: Path) -> None:
    """Normalize both pools and preserve every included row."""
    cfg = config(config_path)
    output = cfg['output']
    schemas = read_yaml(schemas_path)['datasets']
    sites = read_yaml(mapping_path)['datasets']
    pools = {}
    for pool in ('train', 'test'):
        raw = read_csv(output / f'labeled_{pool}_pool.csv')
        validate_source_rows(raw, cfg, pool)
        pools[pool] = [_normalize(row, schemas[row['source_dataset']], sites.get(row['source_dataset'], {})) for row in raw]
    for pool, rows in pools.items():
        if not rows:
            raise ValueError(f'{pool} pool is empty after label harmonization')
        write_csv(output / f'metadata_{pool}_pool.csv', rows)
        report(f'Normalized {pool}', rows)


def to_model_inputs(row: dict[str, Any], age_scale: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-filled numeric values and an observed mask per semantic field."""
    if age_scale <= 0:
        raise ValueError('age_scale must be positive')
    meta = np.zeros(sum(WIDTHS), dtype=np.float32)
    mask = np.zeros(len(FIELDS), dtype=np.float32)
    for i, field in enumerate(FIELDS):
        observed = str(row.get(f'{field}_missing', int(not present(row.get(field))))).lower() in ('0', '0.0', 'false')
        if not observed:
            continue
        value = row[field]
        mask[i] = 1
        if field == 'age':
            meta[SLICES[i]] = float(value) / age_scale
        elif field in ('sex', 'site'):
            categories = SEXES if field == 'sex' else SITES
            meta[SLICES[i].start + categories.index(str(value))] = 1
        else:
            meta[SLICES[i]] = float(value)
    return meta, mask


def _zero_missing(meta: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply semantic masks to all encoded dimensions of a field."""
    result = meta.copy()
    for i, region in enumerate(SLICES):
        result[..., region] *= mask[..., i, None]
    return result, mask.copy()


def apply_random_masking(meta: np.ndarray, mask: np.ndarray, p_field: float,
                         p_all: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Drop only observed fields, reproducibly, including entire metadata records."""
    if not 0 <= p_field <= 1 or not 0 <= p_all <= 1:
        raise ValueError('Masking probabilities must lie in [0, 1]')
    observed = mask.copy() * (rng.random(mask.shape) >= p_field)
    all_drop = rng.random(mask.shape[:-1]) < p_all
    observed = observed * np.expand_dims(~all_drop, -1)
    return _zero_missing(meta, observed)


def get_ablation_mask(mode: str, meta: np.ndarray, mask: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply an ablation without inventing originally unavailable metadata."""
    if mode not in MODES:
        raise ValueError(f'Unknown ablation {mode}; choose {MODES}')
    if mode.startswith('random_'):
        return apply_random_masking(meta, mask, int(mode.split('_')[1]) / 100, 0, rng)
    observed = mask.copy()
    indices = {'drop_age': [0], 'drop_sex': [1], 'drop_site': [2],
               'drop_symptoms': list(range(3, len(FIELDS))), 'all_missing': list(range(len(FIELDS))), 'full': []}
    observed[..., indices[mode]] = 0
    return _zero_missing(meta, observed)
