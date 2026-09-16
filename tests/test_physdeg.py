import random

import torch
from torchvision import transforms as T

from augmentation.physdeg import PhysDegTransform
from degradation.params import ID_TYPES, N_SEVERITIES, PHYSICAL_ORDER
from tests._synthetic import make_synthetic_image

BASE_TRANSFORM = T.Compose([T.Resize((32, 32)), T.PILToTensor(), lambda t: t.float() / 255.0])


def test_output_shape_is_three_views_stacked():
    img = make_synthetic_image()
    tfm = PhysDegTransform(BASE_TRANSFORM, rng=random.Random(0))
    out = tfm(img)
    assert out.shape == (3, 1, 32, 32)  # clean, deg1, deg2


def test_first_view_is_exactly_the_clean_transform():
    img = make_synthetic_image()
    tfm = PhysDegTransform(BASE_TRANSFORM, rng=random.Random(0))
    out = tfm(img)
    assert torch.allclose(out[0], BASE_TRANSFORM(img))


def test_deterministic_given_the_same_rng_state():
    """Fairness requirement from the vNext design docs (U8): the static baseline
    must be exactly reproducible under a fixed training budget/seed so that
    A-PhysDeg vs static PhysDeg is a one-variable comparison, not two runs that
    also differ by sampling noise."""
    img = make_synthetic_image()
    tfm_a = PhysDegTransform(BASE_TRANSFORM, rng=random.Random(123))
    tfm_b = PhysDegTransform(BASE_TRANSFORM, rng=random.Random(123))
    assert torch.equal(tfm_a(img), tfm_b(img))


def test_chain_length_within_range_and_physical_order():
    tfm = PhysDegTransform(BASE_TRANSFORM, chain_range=(2, 3), rng=random.Random(0))
    for _ in range(50):
        chain = tfm._sample_chain()
        types = [t for t, _ in chain]
        assert 2 <= len(types) <= 3
        assert len(set(types)) == len(types), "no repeated degradation type in one chain"
        assert all(t in ID_TYPES for t in types)
        indices = [PHYSICAL_ORDER.index(t) for t in types]
        assert indices == sorted(indices), "chain must be composed in physical acquisition order"
        severities = [s for _, s in chain]
        assert all(0 <= s < N_SEVERITIES for s in severities)


def test_rejects_unknown_degradation_type():
    import pytest
    with pytest.raises(ValueError):
        PhysDegTransform(BASE_TRANSFORM, deg_types=["motion_blur", "not_a_real_degradation"])
