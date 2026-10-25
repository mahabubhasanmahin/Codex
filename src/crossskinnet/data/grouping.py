"""Explicit identifier scope for patient/lesion grouping, driven by dataset schemas."""
from __future__ import annotations

from typing import Any

from crossskinnet.io import present


def assign_groups(rows: list[dict[str, str]], schemas: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Prefer patient IDs; resolve lesion aliases only within their declared scope.

    Patient-scoped lesion numbers are not global biological identifiers. Requiring
    patient IDs and validating configured filename prefixes prevents silently
    treating a repeated number as the same lesion across unrelated patients.
    Unconfigured/global lesions still fail on conflicting patient associations.
    """
    schemas = schemas or {}
    global_links: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        source = row['source_dataset']
        rule = schemas.get(source, {}).get('grouping', {})
        scope = rule.get('lesion_id_scope', 'global')
        if scope not in ('patient', 'global'):
            raise ValueError(f'{source}: lesion_id_scope must be patient or global')
        patient, lesion = row.get('patient_id'), row.get('lesion_id')
        if scope == 'patient' and present(lesion) and not present(patient):
            raise ValueError(f'{source}: patient-scoped lesion requires patient_id for {row["image_id"]}')
        prefix = rule.get('image_id_prefix_template')
        if prefix and present(patient) and present(lesion):
            if not row['image_id'].startswith(prefix.format(patient_id=patient, lesion_id=lesion)):
                raise ValueError(f'{source}: image_id does not match configured patient/lesion prefix: {row["image_id"]}')
        if scope == 'global' and present(patient) and present(lesion):
            global_links.setdefault((source, lesion), set()).add(patient)
    if any(len(values) > 1 for values in global_links.values()):
        raise ValueError('A global lesion ID is associated with multiple patients; review identifier scope in dataset_schemas.yaml')
    result = []
    for row in rows:
        group = next((row[field] for field in ('patient_id', 'lesion_id', 'image_id') if present(row.get(field))), None)
        if group is None:
            raise ValueError('Cannot determine group_id: patient_id, lesion_id and image_id are missing')
        link = global_links.get((row['source_dataset'], row.get('lesion_id', '')))
        if not present(row.get('patient_id')) and link:
            group = next(iter(link))
        result.append({**row, 'group_id': group})
    return result
