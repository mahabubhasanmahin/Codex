"""Mapping-free dataset feasibility reports; never approve clinical labels automatically."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

from crossskinnet.data.dedup import run as deduplicate
from crossskinnet.data.merge import merge_datasets, read_yaml
from crossskinnet.data.metadata import FIELDS
from crossskinnet.io import config, present, read_csv, report, write_csv


def _group_evidence(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count recorded identifiers separately from image-ID fallback proxies."""
    patients = {r['patient_id'] for r in rows if present(r.get('patient_id'))}
    lesions = {r['lesion_id'] for r in rows if present(r.get('lesion_id'))}
    groups = set()
    fallback_images = 0
    lesion_patients: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if present(row.get('patient_id')) and present(row.get('lesion_id')):
            lesion_patients[row['lesion_id']].add(row['patient_id'])
    for row in rows:
        if present(row.get('patient_id')):
            groups.add(('patient', row['patient_id']))
        elif present(row.get('lesion_id')):
            linked = lesion_patients.get(row['lesion_id'], set())
            if len(linked) == 1:
                groups.add(('patient', next(iter(linked))))
            else:
                groups.add(('lesion', row['lesion_id']))
        else:
            groups.add(('image_fallback', row['image_id']))
            fallback_images += 1
    return dict(recorded_patients=len(patients), recorded_lesions=len(lesions),
                split_group_proxies=len(groups), rows_without_patient_or_lesion=fallback_images,
                lesion_numbers_shared_across_patients=sum(len(values) > 1 for values in lesion_patients.values()))


def _class_counts(raw: list[dict[str, str]], kept: list[dict[str, str]],
                  removed: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Report labels as recorded, avoiding unjustified cross-dataset equivalences."""
    by_label: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    remaining: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_label[(row['source_dataset'], row['original_label'])].append(row)
    for row in kept:
        remaining[(row['source_dataset'], row['original_label'])].append(row)
    exclusions = Counter((r['source_dataset'], r['original_label'], r['removal_reason']) for r in removed)
    results = []
    for (source, label), rows in sorted(by_label.items()):
        post = remaining[(source, label)]
        external = exclusions[(source, label, 'external_overlap_component')]
        evidence = _group_evidence(post)
        results.append(dict(source_dataset=source, original_label=label, raw_images=len(rows),
            removed_external_overlap=external, images_after_external_removal=len(rows) - external,
            removed_internal_duplicates=exclusions[(source, label, 'training_duplicate_component')],
            images_after_all_dedup=len(post), **evidence,
            group_evidence='image_fallback_not_patient_independence' if evidence['rows_without_patient_or_lesion'] else
                           ('recorded_identifiers_not_independently_verified' if post else 'no_remaining_images'),
            review='requires_label_review_and_final_cross_source_group_audit'))
    return results


def _metadata_counts(rows: list[dict[str, str]], schemas: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate schema-provided fields from observed values and missing records."""
    result = []
    for source in sorted({r['source_dataset'] for r in rows}):
        subset = [r for r in rows if r['source_dataset'] == source]
        schema = schemas[source]
        declared = set(schema.get('columns', {})) | set(schema.get('symptoms', {})) | set(schema.get('join', {}).get('columns', {}))
        for field in ('patient_id', 'lesion_id', *FIELDS):
            observed = sum(present(r.get(field)) for r in subset)
            result.append(dict(source_dataset=source, field=field, raw_images=len(subset),
                declared_by_schema=field in declared, observed=observed, missing=len(subset) - observed,
                missing_fraction=(len(subset) - observed) / len(subset)))
    return result


def _label_review(rows: list[dict[str, str]], schemas: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose folder-label limitations without making medical taxonomy decisions."""
    counts = Counter((r['source_dataset'], r['original_label']) for r in rows)
    results = []
    for (source, label), count in sorted(counts.items()):
        folder = schemas[source]['format'] == 'folders'
        mixed_wording = ' and ' in label.lower() or 'other' in label.lower()
        results.append(dict(source_dataset=source, original_label=label, raw_images=count,
            label_origin='folder_name' if folder else 'configured_metadata',
            disease_label_approved=False, mixed_wording_flag=folder and mixed_wording,
            required_action=('Image-level diagnosis or documented folder-label verification required; '
                             'mixed/broad folders cannot be auto-mapped to one disease.' if folder else
                             'Review source diagnosis definitions and cross-dataset equivalence in label_mapping.csv.')))
    return results


def run(config_path: Path, schemas_path: Path) -> Path:
    """Refresh raw ingestion and audit isolated copies; leave downstream pools untouched."""
    cfg = config(config_path)
    print('Refreshing raw merge from configured datasets. This audit does not train a model.', flush=True)
    merge_datasets(config_path, schemas_path)
    output = cfg['output']
    train, external = (read_csv(output / f'raw_{pool}_pool.csv') for pool in ('train', 'test'))
    if not train or not external:
        raise ValueError('Audit requires nonempty development and external pools')
    audit = output / 'audit'
    audit.mkdir(parents=True, exist_ok=True)
    # Isolate pre-taxonomy deduplication from reviewed pipeline intermediates.
    write_csv(audit / 'metadata_train_pool.csv', train)
    write_csv(audit / 'raw_test_pool.csv', external)
    audit_cfg = {'seed': cfg['seed'], 'paths': {'output_root': str(audit.resolve())}, 'dedup': cfg.get('dedup', {})}
    audit_config = audit / 'audit_config.yaml'
    audit_config.write_text(yaml.safe_dump(audit_cfg))
    print('Hashing raw images (first run can take time; later runs reuse the cache).', flush=True)
    deduplicate(audit_config)
    kept = read_csv(audit / 'deduped_train_pool.csv')
    removed = read_csv(audit / 'removed_training_duplicates.csv')
    counts = _class_counts(train, kept, removed)
    schemas = read_yaml(schemas_path)['datasets']
    metadata = _metadata_counts(train + external, schemas)
    labels = _label_review(train + external, schemas)
    write_csv(audit / 'class_feasibility.csv', counts)
    write_csv(audit / 'metadata_availability.csv', metadata)
    write_csv(audit / 'label_review.csv', labels)
    write_csv(audit / 'dermnet_review.csv', [r for r in labels if r['source_dataset'] == 'Dermnet'], list(labels[0]))
    summaries = []
    for source in sorted({r['source_dataset'] for r in train + external}):
        subset = [r for r in train + external if r['source_dataset'] == source]
        post = [r for r in kept if r['source_dataset'] == source]
        is_external = source in ('HAM10000', 'PH2')
        summaries.append(dict(source_dataset=source, pool='external_test' if is_external else 'train',
            raw_images=len(subset), retained_images=len(subset) if is_external else len(post),
            **_group_evidence(subset if is_external else post)))
    write_csv(audit / 'source_summary.csv', summaries)
    lines = ['# Dataset feasibility audit', '',
        'No clinical mappings are inferred or approved by this report. Counts are per ORIGINAL label,',
        'not final unified disease. Review source definitions before choosing the final taxonomy.', '',
        '|Source|Pool|Raw images|Retained images|', '|---|---|---:|---:|']
    lines += [f"|{r['source_dataset']}|{r['pool']}|{r['raw_images']}|{r['retained_images']}|" for r in summaries]
    lines += ['', '## Read the reports', '',
        '- class_feasibility.csv: per-source/original-label counts before and after external/internal removal.',
        '- metadata_availability.csv: schema declarations, observed/missing counts for metadata and IDs.',
        '- label_review.csv / dermnet_review.csv: unresolved disease-label evidence; no automatic approvals.',
        '- duplicates_report.csv / duplicate_pairs.png: representative pHash-component edges to inspect.',
        '', '## Important limits', '',
        'Deduplication uses the configured pHash threshold, including conservative transitive components.',
        'Perceptual matches require visual review; exact SHA256 equality is reported separately.',
        'External images are retained even when their labels would later be excluded from evaluation.',
        'Image fallback groups do NOT establish patient independence. Recorded IDs are not independently verified.',
        'Group counts per label are not additive: a patient may contribute multiple labels.',
        'Cross-source identity collisions and group links must be reviewed before final splitting.',
        'Audit copies are raw and unharmonized, not training-ready datasets. They do not replace stages 02–05.',
        'After reviewing clinical/site mappings, rerun stages 02–05; harmonization can change metadata preference.',
        'The review-wording flag is a text heuristic, not evidence of a single-disease folder when false.']
    (audit / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    with ZipFile(audit / 'audit_reports.zip', 'w', ZIP_DEFLATED) as archive:
        for filename in ('REPORT.md', 'source_summary.csv', 'class_feasibility.csv',
                         'metadata_availability.csv', 'label_review.csv', 'dermnet_review.csv'):
            archive.write(audit / filename, arcname=filename)
    report('Source feasibility', summaries)
    report('Class feasibility', counts)
    print(f'Audit complete: {audit / "REPORT.md"}', flush=True)
    return audit


def overlap_evidence(raw: list[dict[str, str]], pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Count proven external connectivity in reported edges, not unseen image hashes.

    SHA-only counts are lower bounds: the duplicate report retains representative
    edges and can omit other exact-equality edges in the same perceptual bucket.
    """
    result = []
    predicates = [('reported_sha256_edges', lambda row: str(row['exact_sha256']).lower() == 'true'),
                  ('reported_equal_phash_edges', lambda row: int(row['distance']) == 0),
                  ('all_reported_phash_edges', lambda row: True)]
    for method, include in predicates:
        parent: dict[tuple[str, str, str], tuple[str, str, str]] = {}

        def find(node: tuple[str, str, str]) -> tuple[str, str, str]:
            parent.setdefault(node, node)
            current = node
            while parent[current] != current:
                parent[current] = parent[parent[current]]
                current = parent[current]
            return current

        for pair in pairs:
            if include(pair):
                nodes = [(pair[f'source_{side}'], pair[f'image_id_{side}'], pair[f'image_path_{side}']) for side in ('a','b')]
                parent[find(nodes[0])] = find(nodes[1])
        external = {find(node) for node in list(parent) if node[0] in ('HAM10000','PH2')}
        counts: Counter = Counter()
        for row in raw:
            node = (row['source_dataset'], row['image_id'], row['image_path'])
            if node in parent and find(node) in external:
                counts[row['source_dataset']] += 1
        for source in sorted({row['source_dataset'] for row in raw}):
            result.append(dict(source_dataset=source, evidence=method, training_images_connected_to_external=counts[source],
                               interpretation='proven_lower_bound' if method == 'reported_sha256_edges' else 'perceptual_not_byte_equality'))
    return result


def review_overlap(config_path: Path) -> Path:
    """Reuse the existing audit report to expose exact evidence and targeted grids."""
    from crossskinnet.data.dedup import save_duplicate_grid

    cfg = config(config_path)
    audit = cfg['output'] / 'audit'
    raw = read_csv(cfg['output'] / 'raw_train_pool.csv')
    pairs = read_csv(audit / 'duplicates_report.csv')
    evidence = overlap_evidence(raw, pairs)
    write_csv(audit / 'overlap_evidence.csv', evidence)
    for source in sorted({r['source_dataset'] for r in raw} - {'ISIC2018'}):
        selected = [r for r in pairs if r['action'] == 'remove_train_external_overlap'
                    and source in (r['source_a'], r['source_b'])]
        write_csv(audit / f'{source}_overlap_pairs.csv', selected, list(pairs[0]) if pairs else
            ['image_id_a','image_id_b','image_path_a','image_path_b','source_a','source_b','distance','exact_sha256','action'])
        save_duplicate_grid(selected, audit / f'{source}_overlap_review.png', int(cfg['seed']), count=20)
        print(f'{source}: {len(selected)} representative component edges; grid samples up to 20, CSV contains all.')
    report('Reported overlap evidence', evidence)
    return audit
