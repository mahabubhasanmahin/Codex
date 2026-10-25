"""Cached exact/perceptual duplicate components using a 64-bit Hamming BK-tree."""
from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from crossskinnet.data.metadata import FIELDS
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
        draw.text((0, index * 130 + 108), f"distance={pair['distance']} {pair['action']}", fill='black')
    if not pairs:
        draw.text((5, 5), 'No duplicate pairs found', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(config_path: Path) -> None:
    """Remove training-only duplicate components, giving external data priority."""
    cfg = config(config_path)
    output, options = cfg['output'], cfg.get('dedup', {})
    threshold = int(options.get('hamming_threshold', 5))
    if not 0 <= threshold <= 64:
        raise ValueError('hamming_threshold must lie in [0, 64]')
    train = read_csv(output / 'metadata_train_pool.csv')
    # Even externally excluded diagnoses must protect against training leakage.
    # Label filtering must never shrink the reference set for overlap detection.
    reference = output / 'raw_test_pool.csv'
    external = read_csv(reference if reference.exists() else output / 'metadata_test_pool.csv')
    if any(r['source_dataset'] in ('HAM10000', 'PH2') for r in train):
        raise ValueError('External data found in training pool')
    rows = train + external
    hashes = cached_hashes(rows, output / 'image_hashes.csv', int(options.get('workers', 4)))
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = []
    tree = BKTree()
    identical: dict[int, int] = {}
    for index, row in enumerate(rows):
        value = int(hashes[row['image_path']]['phash'], 16)
        neighbors = [(identical[value], 0)] if value in identical else tree.query(value, threshold)
        for previous, distance in neighbors:
            left, right = find(index), find(previous)
            if left != right:
                parent[left] = right
            edges.append((previous, index, distance))
        if value not in identical:
            identical[value] = index
            tree.add(value, index)
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    removed: set[int] = set()
    for members in components.values():
        training = [index for index in members if index < len(train)]
        if any(index >= len(train) for index in members):
            removed.update(training)
        elif len(training) > 1:
            best = max(training, key=lambda i: (sum(present(rows[i].get(f)) for f in FIELDS), -i))
            removed.update(set(training) - {best})
    external_roots = {find(i) for i in range(len(train), len(rows))}
    pairs = []
    for left, right, distance in edges:
        a, b = rows[left], rows[right]
        external_component = find(left) in external_roots
        action = 'remove_train_external_overlap' if external_component and (left < len(train) or right < len(train)) else (
            'keep_external' if left >= len(train) and right >= len(train) else 'keep_best_training_metadata')
        pairs.append(dict(image_id_a=a['image_id'], image_id_b=b['image_id'], image_path_a=a['image_path'],
                          image_path_b=b['image_path'], source_a=a['source_dataset'], source_b=b['source_dataset'],
                          distance=distance, exact_sha256=hashes[a['image_path']]['sha256'] == hashes[b['image_path']]['sha256'],
                          action=action))
    kept = [r for i, r in enumerate(train) if i not in removed]
    write_csv(output / 'deduped_train_pool.csv', kept, list(train[0]) if train else [])
    write_csv(output / 'duplicates_report.csv', pairs, ['image_id_a', 'image_id_b', 'image_path_a', 'image_path_b',
              'source_a', 'source_b', 'distance', 'exact_sha256', 'action'])
    removed_rows = [{**train[i], 'removal_reason': 'duplicate_component'} for i in sorted(removed)]
    write_csv(output / 'removed_training_duplicates.csv', removed_rows, [*train[0], 'removal_reason'])
    ham_groups = [dict(image_id=r['image_id'], lesion_id=r.get('lesion_id', 'NaN')) for r in external if r['source_dataset'] == 'HAM10000']
    write_csv(output / 'ham_lesion_groups.csv', ham_groups, ['image_id', 'lesion_id'])
    save_duplicate_grid(pairs, output / 'duplicate_pairs.png', int(cfg['seed']))
    print('Removed training rows by source:', dict(Counter(train[i]['source_dataset'] for i in removed)))
    print('External pool unchanged:', len(external))
    report('Deduplicated training', kept)
