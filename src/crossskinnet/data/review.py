"""Prepare selected-label duplicate candidates and persist individual review decisions."""
from __future__ import annotations

import hashlib
import json
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from crossskinnet.data.dedup import _reviewed, run as deduplicate
from crossskinnet.data.labels import apply as apply_labels
from crossskinnet.io import config, read_csv, write_csv


def _digest(path: Path) -> str:
    """Fingerprint a configuration/input file without interpreting its content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path: Path, mapping_path: Path) -> Path:
    """Filter clinical labels, then preview dedup with ALL raw external references.

    This does not need site normalization, never declares normalized data ready,
    and never assigns a human duplicate decision. Actual stages 03/04 remain
    required before building or training.
    """
    cfg = config(config_path)
    apply_labels(config_path, mapping_path)
    output = cfg['output']
    folder = output / 'review_preparation'
    folder.mkdir(parents=True, exist_ok=True)
    train = read_csv(output / 'labeled_train_pool.csv')
    external = read_csv(output / 'raw_test_pool.csv')
    if not train or not external:
        raise ValueError('Review preparation needs selected training rows and raw external references')
    write_csv(folder / 'metadata_train_pool.csv', train)
    write_csv(folder / 'raw_test_pool.csv', external)
    options = dict(cfg.get('dedup', {}))
    if options.get('grouping_schemas'):
        schema_file = Path(options['grouping_schemas'])
        if not schema_file.is_absolute():
            schema_file = config_path.resolve().parent / schema_file
        options['grouping_schemas'] = str(schema_file)
    reviews = Path(options.get('reviews_csv') or output / 'near_duplicate_reviews.csv')
    if not reviews.is_absolute():
        reviews = config_path.resolve().parent / reviews
    options['reviews_csv'] = str(reviews.resolve())
    preview = dict(cfg)
    preview.pop('output', None)
    preview['paths'] = {**cfg['paths'], 'output_root':str(folder.resolve())}
    preview['dedup'] = options
    preview['review_preview_only'] = True
    path = folder / 'preview_config.yaml'
    path.write_text(yaml.safe_dump(preview))
    cache = folder / 'image_hashes.csv'
    if not cache.exists():
        for prior in (output / 'image_hashes.csv',output / 'audit/image_hashes.csv'):
            if prior.exists():
                shutil.copyfile(prior,cache)
                break
    deduplicate(path, allow_pending=True)
    pairs = read_csv(folder / 'duplicates_report.csv')
    active = {r['pair_id'] for r in pairs if r['action']=='review_pending'}
    _, records = _reviewed(reviews)
    queue = [records[identity] for identity in sorted(active)]
    fields = ['pair_id','sha256_a','sha256_b','source_a','source_b','image_id_a','image_id_b',
              'image_path_a','image_path_b','distance','is_duplicate','notes']
    write_csv(folder / 'review_queue.csv', queue, fields)
    summary = dict(selected_train_rows=len(train), labeled_external_rows=len(read_csv(output / 'labeled_test_pool.csv')),
        raw_external_reference_rows=len(external), pending_relevant_pairs=len(queue),
        verified_only_provisional_train_rows=len(read_csv(folder / 'provisional_train_pool.csv')),
        normalized_training_ready=False)
    for name, rows in [('selected_class_counts', train),
                       ('provisional_class_counts',read_csv(folder / 'provisional_train_pool.csv'))]:
        counts = Counter((row['source_dataset'],row['unified_label']) for row in rows)
        write_csv(folder / f'{name}.csv',[dict(source_dataset=source,unified_label=label,count=count)
            for (source,label),count in sorted(counts.items())],['source_dataset','unified_label','count'])
    (folder / 'selection_summary.json').write_text(json.dumps(summary, indent=2))
    files = [config_path.resolve(),mapping_path.resolve(),output / 'raw_train_pool.csv',output / 'raw_test_pool.csv']
    manifest = dict(files={str(file.resolve()):_digest(file) for file in files},
        reviews_csv=str(reviews.resolve()),queue_sha256=_digest(folder / 'review_queue.csv'))
    (folder / 'review_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Selected-label review preparation:',summary, flush=True)
    print('Preview only; site normalization and actual stage 04 are still required.', flush=True)
    return folder


class ReviewSession:
    """Review one active pair at a time; preserve unrelated content-bound decisions."""

    def __init__(self, config_path: Path) -> None:
        cfg = config(config_path)
        self.folder = cfg['output'] / 'review_preparation'
        manifest = json.loads((self.folder / 'review_manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if _digest(Path(name)) != digest:
                raise ValueError(f'Review input changed: {name}; rerun 04_prepare_review.py')
        if _digest(self.folder / 'review_queue.csv') != manifest['queue_sha256']:
            raise ValueError('Review queue changed; rerun preparation instead of editing it')
        self.reviews_path = Path(manifest['reviews_csv'])
        self.rows = read_csv(self.folder / 'review_queue.csv')
        self.ids = {row['pair_id'] for row in self.rows}

    def decisions(self) -> dict[str, str]:
        """Always read current decisions to avoid overwriting unrelated edits."""
        decisions, _ = _reviewed(self.reviews_path)
        return decisions

    def save(self, pair_id: str, decision: bool | None, notes: str) -> None:
        """Save an explicit per-pair decision; None keeps uncertainty unresolved."""
        if pair_id not in self.ids:
            raise ValueError('Pair is not in the prepared relevant review queue')
        if decision is not None and not isinstance(decision,bool):
            raise ValueError('Decision must be True, False or None')
        _, records = _reviewed(self.reviews_path)
        if pair_id not in records:
            raise ValueError('Review record disappeared; rerun preparation')
        records[pair_id]['is_duplicate'] = '' if decision is None else str(decision)
        records[pair_id]['notes'] = notes.strip()
        with tempfile.NamedTemporaryFile(dir=self.reviews_path.parent, suffix='.csv', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            write_csv(temporary,list(records.values()),list(next(iter(records.values()))))
            temporary.replace(self.reviews_path)
        finally:
            temporary.unlink(missing_ok=True)
