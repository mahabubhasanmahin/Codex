"""Cached exact/perceptual duplicate components using a 64-bit Hamming BK-tree."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from crossskinnet.data.metadata import FIELDS
from crossskinnet.data.sources import validate_source_rows
from crossskinnet.io import config, present, read_csv, report, write_csv


class BKTree:
    """Index distinct 64-bit hashes; repeated hashes use a representative node."""

    def __init__(self) -> None:
        self.nodes: list[tuple[int, int, dict[int, int]]] = []

    def add(self, value: int, index: int) -> None:
        """Insert a previously unseen hash."""
        if not self.nodes:
            self.nodes.append((value, index, {}))
            return
        current = 0
        while True:
            other, _, children = self.nodes[current]
            distance = (value ^ other).bit_count()
            if not distance:
                return
            if distance not in children:
                children[distance] = len(self.nodes)
                self.nodes.append((value, index, {}))
                return
            current = children[distance]

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        """Find representative neighbors without a full Python pairwise scan."""
        stack = [0] if self.nodes else []
        result = []
        while stack:
            other, index, children = self.nodes[stack.pop()]
            distance = (value ^ other).bit_count()
            if distance <= radius:
                result.append((index, distance))
            stack.extend(child for edge, child in children.items() if distance - radius <= edge <= distance + radius)
        return result


def _hash_file(path: str) -> dict[str, Any]:
    """Compute hashes while retaining file fingerprint evidence for cache refresh."""
    file = Path(path)
    stat = file.stat()
    with file.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()
    with Image.open(file) as image:
        phash = str(imagehash.phash(image.convert('RGB'), hash_size=8))
    return dict(image_path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest, phash=phash)


def cached_hashes(rows: list[dict[str, str]], cache: Path, workers: int = 4) -> dict[str, dict[str, Any]]:
    """Reuse hashes only when path, size and modification timestamp still match."""
    existing = {r['image_path']: r for r in read_csv(cache)} if cache.exists() else {}
    result = {}
    pending = []
    for path in dict.fromkeys(r['image_path'] for r in rows):
        stat = Path(path).stat()
        old = existing.get(path)
        if old and str(stat.st_size) == str(old['size']) and str(stat.st_mtime_ns) == str(old['mtime_ns']):
            result[path] = old
        else:
            pending.append(path)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(_hash_file, pending):
            result[row['image_path']] = row
    write_csv(cache, list(result.values()), ['image_path', 'size', 'mtime_ns', 'sha256', 'phash'])
    return result


def save_duplicate_grid(pairs: list[dict[str, Any]], path: Path, seed: int, count: int = 20) -> None:
    """Save a reproducible sample of duplicate edges for visual auditing."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pairs), min(count, len(pairs)), replace=False) if pairs else []
    canvas = Image.new('RGB', (360, max(1, len(chosen)) * 130), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, selected in enumerate(chosen):
        pair = pairs[int(selected)]
        for column, field in enumerate(('image_path_a', 'image_path_b')):
            with Image.open(pair[field]) as image:
                tile = image.convert('RGB')
                tile.thumbnail((175, 105))
                canvas.paste(tile, (column * 180, index * 130))
        draw.text((0, index * 130 + 108), f"d={pair['distance']} id={str(pair.get('pair_id',''))[:8]} {pair['action']}", fill='black')
    if not pairs:
        draw.text((5, 5), 'No duplicate pairs found', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)



def _pair_id(first: str, second: str) -> str:
    """Bind a review decision to immutable file content, independent of path/order."""
    return hashlib.sha256('|'.join(sorted((first, second))).encode()).hexdigest()


def _reviewed(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Read explicit duplicate decisions, rejecting malformed or conflicting edits."""
    decisions, records = {}, {}
    for row in read_csv(path) if path.exists() else []:
        if row['pair_id'] != _pair_id(row['sha256_a'], row['sha256_b']):
            raise ValueError(f'Invalid content-bound review pair_id in {path}')
        decision = row['is_duplicate'].strip().lower()
        if decision not in ('', 'true', 'false'):
            raise ValueError(f'is_duplicate must be blank, True or False in {path}')
        if row['pair_id'] in records:
            raise ValueError(f'Duplicate review pair_id in {path}')
        records[row['pair_id']] = row
        decisions[row['pair_id']] = decision
    return decisions, records


def _candidate_edges(rows: list[dict[str, str]], hashes: dict[str, dict[str, Any]],
                     threshold: int, train_count: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Index one representative per SHA; query pHash neighbors output-sensitively.

    Exact grouping is independent of pHash buckets. Within a collision bucket,
    all distinct SHA pairs are candidates, including pairs hidden by a rejected
    bucket representative. External-only SHA clusters do not require deduplication.
    """
    by_sha: dict[str, int] = {}
    exact, candidates = [], []
    for index, row in enumerate(rows):
        digest = str(hashes[row['image_path']]['sha256'])
        if digest in by_sha:
            other = by_sha[digest]
            distance = (int(str(hashes[row['image_path']]['phash']),16) ^
                        int(str(hashes[rows[other]['image_path']]['phash']),16)).bit_count()
            exact.append((other, index, distance))
        else:
            by_sha[digest] = index
    buckets: dict[int, list[int]] = {}
    tree = BKTree()
    hash_values: list[int] = []
    for index in by_sha.values():
        value = int(str(hashes[rows[index]['image_path']]['phash']), 16)
        for bucket_index, distance in tree.query(value, threshold):
            for other in buckets[hash_values[bucket_index]]:
                if index < train_count or other < train_count:
                    candidates.append((other, index, distance))
        if value not in buckets:
            tree.add(value, len(hash_values))
            hash_values.append(value)
            buckets[value] = []
        buckets[value].append(index)
    return exact, candidates


def run(config_path: Path, allow_pending: bool = False) -> None:
    """Remove SHA-exact/review-confirmed duplicates; gate readiness on pending review.

    allow_pending is for audit previews only. It never marks a pool ready or
    publishes an unreviewed deduped_train_pool.csv. All external images remain
    references, including diagnoses excluded by harmonization.
    """
    cfg = config(config_path)
    output, options = cfg['output'], cfg.get('dedup', {})
    threshold = int(options.get('hamming_threshold', 5))
    if not 0 <= threshold <= 64:
        raise ValueError('hamming_threshold must lie in [0, 64]')
    if options.get('near_duplicate_policy', 'review') != 'review':
        raise ValueError('near_duplicate_policy must be review; pHash alone cannot confirm duplicates')
    train = read_csv(output / 'metadata_train_pool.csv')
    reference = output / 'raw_test_pool.csv'
    external = read_csv(reference if reference.exists() else output / 'metadata_test_pool.csv')
    if not train:
        raise ValueError('Training pool is empty')
    if any(r['source_dataset'] in ('HAM10000', 'PH2') for r in train):
        raise ValueError('External data found in training pool')
    validate_source_rows(train, cfg, 'train')
    validate_source_rows(external, cfg, 'test')
    rows = train + external
    hashes = cached_hashes(rows, output / 'image_hashes.csv', int(options.get('workers', 4)))
    exact, candidates = _candidate_edges(rows, hashes, threshold, len(train))
    reviews = Path(options.get('reviews_csv') or output / 'near_duplicate_reviews.csv')
    if not reviews.is_absolute():
        reviews = config_path.resolve().parent / reviews
    decisions, review_records = _reviewed(reviews)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    for left, right, _ in exact:
        union(left, right)
    pair_ids = {}
    for left, right, distance in candidates:
        a, b = rows[left], rows[right]
        sha_a, sha_b = str(hashes[a['image_path']]['sha256']), str(hashes[b['image_path']]['sha256'])
        identity = _pair_id(sha_a, sha_b)
        pair_ids[(left, right)] = identity
        previous_review = review_records.get(identity, {})
        review_records[identity] = dict(pair_id=identity, sha256_a=sha_a, sha256_b=sha_b,
                source_a=a['source_dataset'], source_b=b['source_dataset'], image_id_a=a['image_id'],
                image_id_b=b['image_id'], image_path_a=a['image_path'], image_path_b=b['image_path'],
                distance=str(distance), is_duplicate=previous_review.get('is_duplicate',''), notes=previous_review.get('notes',''))
        if decisions.get(identity) == 'true':
            union(left, right)
    review_fields = ['pair_id','sha256_a','sha256_b','source_a','source_b','image_id_a','image_id_b',
                     'image_path_a','image_path_b','distance','is_duplicate','notes']
    write_csv(reviews, list(review_records.values()), review_fields)
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    external_roots = {find(index) for index in range(len(train), len(rows))}
    removed: set[int] = set()
    for root, members in components.items():
        training = [index for index in members if index < len(train)]
        if root in external_roots:
            removed.update(training)
        elif len(training) > 1:
            best = max(training, key=lambda i: (sum(present(rows[i].get(field)) for field in FIELDS), -i))
            removed.update(set(training) - {best})
    kept = [row for index, row in enumerate(train) if index not in removed]
    retained_roots = {find(index) for index in range(len(train)) if index not in removed}
    pending, conflicting = [], []
    pairs = []
    for left, right, distance in exact + candidates:
        a, b = rows[left], rows[right]
        is_exact = str(hashes[a['image_path']]['sha256']) == str(hashes[b['image_path']]['sha256'])
        identity = '' if is_exact else pair_ids[(left,right)]
        decision = 'true' if is_exact else decisions.get(identity, '')
        accepted = is_exact or decision == 'true'
        relevant = find(left) in retained_roots or find(right) in retained_roots
        if not accepted:
            if decision == 'false':
                action = 'review_rejected_not_duplicate'
                if find(left) == find(right):
                    conflicting.append(identity)
            elif relevant:
                action = 'review_pending'
                pending.append(identity)
            else:
                action = 'review_pending_no_retained_training_member'
        elif find(left) in external_roots and (left < len(train) or right < len(train)):
            action = 'remove_train_external_overlap'
        else:
            action = 'keep_external' if left >= len(train) and right >= len(train) else 'keep_best_training_metadata'
        pairs.append(dict(image_id_a=a['image_id'],image_id_b=b['image_id'],image_path_a=a['image_path'],
            image_path_b=b['image_path'],source_a=a['source_dataset'],source_b=b['source_dataset'],distance=distance,
            exact_sha256=is_exact, pair_id=identity, decision_basis='sha256' if is_exact else 'human_review' if decision else 'phash_candidate', action=action))
    write_csv(output / 'duplicates_report.csv', pairs, ['image_id_a','image_id_b','image_path_a','image_path_b',
        'source_a','source_b','distance','exact_sha256','pair_id','decision_basis','action'])
    write_csv(output / 'provisional_train_pool.csv', kept, list(train[0]))
    ready = not pending and not conflicting
    if ready:
        write_csv(output / 'deduped_train_pool.csv', kept, list(train[0]))
    status = dict(policy='review', ready=ready, pending_relevant_pairs=len(set(pending)),
        conflicting_reviews=len(set(conflicting)), reviews_csv=str(reviews),
        reviews_sha256=hashlib.sha256(reviews.read_bytes()).hexdigest(), hamming_threshold=threshold,
        train_pool_sha256=hashlib.sha256((output / 'deduped_train_pool.csv').read_bytes()).hexdigest() if ready else None)
    (output / 'dedup_status.json').write_text(json.dumps(status, indent=2))
    removed_rows = [{**train[index], 'removal_reason': 'external_overlap_component' if find(index) in external_roots
                     else 'training_duplicate_component'} for index in sorted(removed)]
    write_csv(output / 'removed_training_duplicates.csv', removed_rows, [*train[0],'removal_reason'])
    write_csv(output / 'ham_lesion_groups.csv', [dict(image_id=r['image_id'],lesion_id=r.get('lesion_id','NaN'))
        for r in external if r['source_dataset']=='HAM10000'], ['image_id','lesion_id'])
    save_duplicate_grid(pairs, output / 'duplicate_pairs.png', int(cfg['seed']))
    save_duplicate_grid([pair for pair in pairs if pair['action']=='review_pending'],
                        output / 'near_duplicate_review.png', int(cfg['seed']))
    print('Verified removals by training source:', dict(Counter(train[index]['source_dataset'] for index in removed)))
    print('External pool unchanged:', len(external), '; relevant pending reviews:', len(set(pending)))
    report('Provisional training (not ready when reviews remain)', kept)
    if conflicting:
        raise ValueError(f'Conflicting duplicate reviews inside accepted components: {len(set(conflicting))}; correct {reviews}')
    if pending and not allow_pending:
        raise ValueError(f'{len(set(pending))} pHash candidate pairs require review in {reviews}. '
                         'Fill is_duplicate=True/False, rerun dedup, then build. No training-ready pool was published.')
