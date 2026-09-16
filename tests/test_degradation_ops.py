import numpy as np
import pytest

from degradation.ops import ID_OPS, OOD_OPS
from degradation.params import ID_TYPES, N_SEVERITIES, OOD_FAMILIES, OOD_TYPES
from tests._synthetic import make_synthetic_image

STOCHASTIC_ID = {"gaussian_noise"}
STOCHASTIC_OOD = {"glass_blur", "speckle_noise", "impulse_noise"}


def test_seven_id_eight_ood_no_overlap():
    assert len(ID_TYPES) == 7
    assert len(OOD_TYPES) == 8
    assert set(ID_TYPES).isdisjoint(OOD_TYPES), "ID and OOD sets must not overlap"
    assert set(ID_OPS) == set(ID_TYPES)
    assert set(OOD_OPS) == set(OOD_TYPES)


def test_ood_families_partition_all_eight_types():
    members = sorted(t for fam in OOD_FAMILIES.values() for t in fam)
    assert members == sorted(OOD_TYPES)


@pytest.mark.parametrize("deg_type", ID_TYPES)
def test_id_ops_preserve_shape_and_mode(deg_type):
    img = make_synthetic_image()
    for s in range(N_SEVERITIES):
        out = ID_OPS[deg_type](img, s, seed=42)
        assert out.size == img.size
        assert out.mode == "L"


@pytest.mark.parametrize("deg_type", OOD_TYPES)
def test_ood_ops_preserve_shape_and_mode(deg_type):
    img = make_synthetic_image()
    for s in range(N_SEVERITIES):
        out = OOD_OPS[deg_type](img, s, seed=42)
        assert out.size == img.size
        assert out.mode == "L"


@pytest.mark.parametrize("deg_type", sorted(STOCHASTIC_ID))
def test_id_stochastic_ops_are_seed_deterministic(deg_type):
    img = make_synthetic_image()
    a = np.asarray(ID_OPS[deg_type](img, 1, seed=7))
    b = np.asarray(ID_OPS[deg_type](img, 1, seed=7))
    c = np.asarray(ID_OPS[deg_type](img, 1, seed=8))
    assert np.array_equal(a, b), "same seed must reproduce the exact same corruption"
    assert not np.array_equal(a, c), "different seeds should (almost certainly) differ"


@pytest.mark.parametrize("deg_type", sorted(STOCHASTIC_OOD))
def test_ood_stochastic_ops_are_seed_deterministic(deg_type):
    img = make_synthetic_image()
    a = np.asarray(OOD_OPS[deg_type](img, 1, seed=7))
    b = np.asarray(OOD_OPS[deg_type](img, 1, seed=7))
    c = np.asarray(OOD_OPS[deg_type](img, 1, seed=8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_severity_is_monotonically_harsher_for_photometric():
    """Sanity check on the severity tables themselves, not just the code:
    brightness/contrast/low_light/gamma should move further from identity as
    severity increases (this is what "severity" is supposed to mean)."""
    img = make_synthetic_image(mean=128.0)
    base_mean = np.asarray(img, dtype=np.float32).mean()

    bright_devs = [abs(np.asarray(ID_OPS["brightness"](img, s)).mean() - base_mean)
                   for s in range(N_SEVERITIES)]
    assert bright_devs == sorted(bright_devs)

    contrast_devs = [np.asarray(ID_OPS["contrast"](img, s)).std() for s in range(N_SEVERITIES)]
    assert contrast_devs == sorted(contrast_devs, reverse=True)  # heavier severity flattens more

    lowlight_means = [np.asarray(OOD_OPS["low_light"](img, s)).mean() for s in range(N_SEVERITIES)]
    assert lowlight_means == sorted(lowlight_means, reverse=True)  # heavier severity = darker


def test_jpeg_and_downsample_reduce_high_frequency_energy():
    """Lossy ops should not increase high-frequency detail; use gradient energy
    as a coarse high-frequency proxy."""
    rng_img = make_synthetic_image(mean=128.0, seed=1)
    base = np.asarray(rng_img, dtype=np.float32)
    base_energy = np.abs(np.diff(base, axis=0)).sum() + np.abs(np.diff(base, axis=1)).sum()

    for deg_type in ("jpeg", "downsample", "motion_blur", "defocus_blur"):
        out = np.asarray(ID_OPS[deg_type](rng_img, N_SEVERITIES - 1), dtype=np.float32)
        out_energy = np.abs(np.diff(out, axis=0)).sum() + np.abs(np.diff(out, axis=1)).sum()
        assert out_energy <= base_energy, f"{deg_type} at max severity should not sharpen the image"
