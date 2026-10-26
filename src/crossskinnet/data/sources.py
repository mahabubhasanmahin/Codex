"""Configured study roles, with PH2 always held out and legacy protocols explicit."""
from __future__ import annotations

from typing import Any

KNOWN_SOURCES = ('PAD-UFES-20', 'ISIC2018', 'Dermnet', 'HAM10000', 'PH2')


def enabled_sources(cfg: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Read explicit roles; PH2 is always external and never optional.

    HAM may be development in the current protocol or external in an explicitly
    configured historical protocol. Required training sources bind each study
    to its intended design, preventing a silent role change in the current one.
    """
    datasets = cfg.get('datasets')
    if datasets is None:
        return {'PAD-UFES-20', 'HAM10000', 'Dermnet'}, {'PH2'}
    unknown = set(datasets) - set(KNOWN_SOURCES)
    if unknown:
        raise ValueError(f'Unknown dataset names: {sorted(unknown)}')
    if 'PH2' not in datasets:
        raise ValueError('External reference PH2 must remain configured')
    training, external = set(), set()
    for name, settings in datasets.items():
        enabled = settings.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: enabled must be a YAML boolean')
        pool = settings.get('pool')
        if pool not in ('train', 'external_test'):
            raise ValueError(f'{name}: pool must be train or external_test')
        if name == 'PH2' and pool != 'external_test':
            raise ValueError('PH2: pool must be external_test; PH2 cannot train/validate')
        if name in ('PAD-UFES-20','ISIC2018','Dermnet') and pool != 'train':
            raise ValueError(f'{name}: pool must be train in the supported protocols')
        if pool == 'external_test' and not enabled:
            raise ValueError(f'{name}: external overlap references cannot be disabled')
        if enabled:
            (training if pool == 'train' else external).add(name)
    if not training:
        raise ValueError('Enable at least one development dataset')
    required = set(cfg.get('study', {}).get('required_training_sources', []))
    if required - training:
        raise ValueError(f'Required training sources are disabled or assigned external: {sorted(required - training)}')
    return training, external


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


def validate_target_labels(rows: list[dict[str, Any]], cfg: dict[str, Any], context: str) -> None:
    """Keep a declared class task fixed rather than silently losing a class."""
    target = cfg.get('study', {}).get('target_labels')
    if target is None:
        return
    if not isinstance(target, list) or not target or any(not isinstance(label,str) or not label.strip() for label in target):
        raise ValueError('study.target_labels must be a nonempty list of label names')
    if len(set(target)) != len(target):
        raise ValueError('study.target_labels contains duplicate names')
    actual = {r['unified_label'] for r in rows}
    if actual != set(target):
        raise ValueError(f'{context}: target label mismatch; missing={sorted(set(target)-actual)}, '
                         f'unexpected={sorted(actual-set(target))}. Review mappings/data before training.')
