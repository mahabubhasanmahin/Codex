"""Field normalization and reproducible mask protocols."""
import numpy as np
import pytest

from tests.test_merge import REPO
from crossskinnet.data.metadata import FIELDS, WIDTHS, _normalize, apply_random_masking, get_ablation_mask, to_model_inputs


def test_normalization_and_structural_masks():
    """Unknowns remain missing, false symptoms remain observed and IDs are untouched."""
    row = dict(image_id='x', source_dataset='PAD-UFES-20', age='50', sex='FEMALE', site='ARM', itch='false')
    schema = {'columns': {'age': 'age', 'sex': 'sex', 'site': 'site'}, 'symptoms': {'itch': 'itch'}}
    normalized = _normalize(row, schema, {'ARM': 'upper_limb'})
    assert normalized['age'] == 50.0 and normalized['sex'] == 'female'
    assert normalized['itch'] == 0 and normalized['itch_missing'] == 0
    assert normalized['grew_missing'] == normalized['structurally_missing_grew'] == 1
    meta, mask = to_model_inputs(normalized)
    assert meta[0] == .5 and mask.sum() == 4
    with pytest.raises(ValueError, match='Map raw site'):
        _normalize(row, schema, {})


def test_masking_reproducible_and_all_missing():
    """Missing dimensions stay zero and all-missing erases every dimension."""
    meta = np.ones((8, sum(WIDTHS)), dtype=np.float32)
    mask = np.ones((8, len(FIELDS)), dtype=np.float32)
    first = apply_random_masking(meta, mask, .5, .3, np.random.default_rng(42))
    second = apply_random_masking(meta, mask, .5, .3, np.random.default_rng(42))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    zero, missing = get_ablation_mask('all_missing', meta, mask, np.random.default_rng(42))
    assert not zero.any() and not missing.any()
    for mode in ('full', 'drop_age', 'drop_sex', 'drop_site', 'drop_symptoms', 'random_25', 'random_50', 'random_75'):
        values, observed = get_ablation_mask(mode, *first, np.random.default_rng(42))
        assert (observed <= first[1]).all()
        assert values.shape == meta.shape
