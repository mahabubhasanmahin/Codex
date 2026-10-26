"""Configured source selection with immutable external-test roles."""
from __future__ import annotations

from typing import Any

TRAIN_SOURCES = ('PAD-UFES-20', 'ISIC2018', 'Dermnet')
EXTERNAL_SOURCES = ('HAM10000', 'PH2')


def enabled_sources(cfg: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Honor explicit enabled flags; external references can never be disabled."""
    datasets = cfg.get('datasets')
    if datasets is None:
        return set(TRAIN_SOURCES), set(EXTERNAL_SOURCES)
    unknown = set(datasets) - set(TRAIN_SOURCES + EXTERNAL_SOURCES)
    if unknown:
        raise ValueError(f'Unknown dataset names: {sorted(unknown)}')
    active_train = set()
    for name in TRAIN_SOURCES + EXTERNAL_SOURCES:
        settings = datasets.get(name)
        if settings is None:
            if name in EXTERNAL_SOURCES:
                raise ValueError(f'External reference {name} must remain configured')
            continue
        enabled = settings.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: enabled must be a YAML boolean')
        expected = 'train' if name in TRAIN_SOURCES else 'external_test'
        if settings.get('pool') != expected:
            raise ValueError(f'{name}: pool must be {expected}; external data cannot train/validate')
        if name in EXTERNAL_SOURCES and not enabled:
            raise ValueError(f'{name}: external overlap references cannot be disabled')
        if enabled and name in TRAIN_SOURCES:
            active_train.add(name)
    if not active_train:
        raise ValueError('Enable at least one development dataset')
    return active_train, set(EXTERNAL_SOURCES)


def validate_source_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], pool: str) -> None:
    """Reject stale tables that include disabled sources instead of silently filtering."""
    training, external = enabled_sources(cfg)
    allowed = external if pool in ('test', 'external_test', 'zero_shot') else training
    unexpected = {r['source_dataset'] for r in rows} - allowed
    if unexpected:
        raise ValueError(f'{pool}: disabled or wrong-pool sources {sorted(unexpected)}; '
                         'rerun merge and downstream stages with the current source configuration')


def validate_training_coverage(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    """Do not label a run as a two-source experiment when one source disappeared."""
    required = set(cfg.get('study', {}).get('required_training_sources', []))
    enabled, _ = enabled_sources(cfg)
    if required - enabled:
        raise ValueError(f'Required training sources are disabled: {sorted(required - enabled)}')
    present_sources = {r['source_dataset'] for r in rows}
    missing = required - present_sources
    if missing:
        raise ValueError(f'Required training sources absent after label/dedup/split: {sorted(missing)}. '
                         'Review dataset availability and labels; do not claim an unrun source.')
