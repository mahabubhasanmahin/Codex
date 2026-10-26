"""Exclude uncertain development groups without declaring their images duplicates."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from crossskinnet.data.grouping import assign_groups


def quarantine_groups(train: list[dict[str, str]], roots: list[int], external_roots: set[int],
                      pending_edges: list[tuple[int, int]], schemas: dict[str, Any]) -> tuple[set[int], dict[int, str]]:
    """Close uncertain/externally-linked exclusions over groups and verified copies.

    `roots` are VERIFIED duplicate-component roots for train+external rows.
    Group connections here never become duplicate verdicts. PH2/external indices
    cannot be quarantined; only development indices are returned.
    """
    grouped = assign_groups(train, schemas)
    parent = list(range(len(train)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        parent[find(first)] = find(second)

    groups, copies = {}, {}
    members: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(grouped):
        key = (row['source_dataset'],row['group_id'])
        if key in groups:
            union(index,groups[key])
        groups[key] = index
        if roots[index] in copies:
            union(index,copies[roots[index]])
        copies[roots[index]] = index
        members[roots[index]].append(index)
    seeds: dict[int, set[str]] = defaultdict(set)
    for root in external_roots:
        for index in members.get(root,[]):
            seeds[find(index)].add('external_linked_group')
    for first, second in pending_edges:
        for endpoint in (first,second):
            for index in members.get(roots[endpoint],[]):
                seeds[find(index)].add('uncertain_candidate_group')
    # All unions precede seed resolution; each component is closed in one pass.
    excluded = {index for index in range(len(train)) if find(index) in seeds}
    reasons = {index:'|'.join(sorted(seeds[find(index)])) for index in excluded}
    return excluded, reasons
