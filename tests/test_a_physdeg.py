import random
from collections import Counter

from torchvision import transforms as T

from augmentation.a_physdeg import AdaptivePhysDegController, APhysDegTransform, weighted_sample_chain
from augmentation.physdeg import PhysDegTransform
from degradation.params import ID_TYPES, N_SEVERITIES
from tests._synthetic import make_synthetic_image, make_synthetic_samples

BASE_TRANSFORM = T.Compose([T.Resize((16, 16)), T.PILToTensor(), lambda t: t.float() / 255.0])


def _stub_difficulty_fn(images, labels):
    """Deterministic stand-in for a real model's eval loop: 'difficulty' = mean
    pixel value, just so different (deg_type, severity) cells get different,
    reproducible numbers to drive the EMA with."""
    return float(images.mean())


# ---- curriculum gating on cell_distribution ----

def test_cell_distribution_respects_curriculum_at_progress_zero():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(0.0)
    dist = ctrl.cell_distribution()
    assert set(s for _, s in dist) == {0}
    assert len(dist) == len(ID_TYPES)  # 7 types x severity 0 only
    assert sum(dist.values()) == 1.0 or abs(sum(dist.values()) - 1.0) < 1e-9


def test_cell_distribution_unlocks_all_severities_by_progress_one():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(1.0)
    dist = ctrl.cell_distribution()
    assert len(dist) == len(ID_TYPES) * N_SEVERITIES  # 21


# ---- reduces to component-1's static PhysDeg in the degenerate case ----

def test_reduces_to_static_uniform_when_curriculum_disabled_and_mix_alpha_zero():
    """unlock_fractions=(0,0,0) + mix_alpha=0 => uniform over all 21 cells, the
    same effective distribution as PhysDegTransform's uniform-type/uniform-
    severity draw. Checked empirically over many draws (not exact per-draw
    equality, since the two sampling *algorithms* differ) with a generous
    tolerance for sampling noise."""
    n_draws = 4000
    ctrl = AdaptivePhysDegController(unlock_fractions=(0.0, 0.0, 0.0), mix_alpha=0.0)
    ctrl.set_progress(0.0)

    a_counts = Counter()
    rng_a = random.Random(0)
    for _ in range(n_draws):
        for t, s in weighted_sample_chain(rng_a, ctrl.cell_distribution(), (2, 3), ID_TYPES):
            a_counts[(t, s)] += 1

    static = PhysDegTransform(BASE_TRANSFORM, rng=random.Random(1))
    static_counts = Counter()
    for _ in range(n_draws):
        for t, s in static._sample_chain():
            static_counts[(t, s)] += 1

    total_a = sum(a_counts.values())
    total_static = sum(static_counts.values())
    for t in ID_TYPES:
        for s in range(N_SEVERITIES):
            fa = a_counts[(t, s)] / total_a
            fs = static_counts[(t, s)] / total_static
            assert abs(fa - fs) < 0.01, f"cell {(t, s)}: A-PhysDeg {fa:.4f} vs static {fs:.4f}"


# ---- run_diagnostics ----

def test_run_diagnostics_rejects_non_train_split():
    import pytest
    ctrl = AdaptivePhysDegController()
    samples = make_synthetic_samples(n_per_class=2, n_classes=2, size=(16, 16))
    with pytest.raises(ValueError):
        ctrl.run_diagnostics(samples, BASE_TRANSFORM, _stub_difficulty_fn, batch_size=4, split="test")


def test_run_diagnostics_touches_every_cell_and_updates_ema():
    ctrl = AdaptivePhysDegController(ema_decay=0.5, init_difficulty=0.5)
    samples = make_synthetic_samples(n_per_class=4, n_classes=2, size=(16, 16))
    raw = ctrl.run_diagnostics(samples, BASE_TRANSFORM, _stub_difficulty_fn,
                                batch_size=4, rng=random.Random(0), split="train")
    assert set(raw.keys()) == {(t, s) for t in ID_TYPES for s in range(N_SEVERITIES)}
    # every cell must have moved away from the 0.5 init (mean-pixel difficulty is
    # never exactly 0.5 for these synthetic images)
    for t in ID_TYPES:
        for s in range(N_SEVERITIES):
            assert ctrl.difficulty.get(t, s) != 0.5


def test_higher_measured_difficulty_increases_a_cells_sampling_weight():
    """The whole point of A-PhysDeg: a cell the model currently struggles with
    should become more likely to be sampled, all else equal."""
    ctrl = AdaptivePhysDegController(mix_alpha=1.0, prob_cap=0.5, ema_decay=0.0)
    ctrl.set_progress(1.0)  # all severities unlocked, so all 21 cells compete
    before = ctrl.cell_distribution()[("motion_blur", 1)]

    for _ in range(5):
        ctrl.difficulty.update("motion_blur", 1, 1.0)  # maximally hard, decay=0 => snaps to 1.0

    after = ctrl.cell_distribution()[("motion_blur", 1)]
    assert after > before


def test_prob_cap_holds_even_under_extreme_single_cell_difficulty():
    ctrl = AdaptivePhysDegController(mix_alpha=1.0, prob_cap=0.15, ema_decay=0.0)
    ctrl.set_progress(1.0)
    ctrl.difficulty.update("jpeg", 2, 1.0)  # one cell maxed out, decay=0 => snaps to 1.0
    dist = ctrl.cell_distribution()
    assert all(p <= 0.15 + 1e-9 for p in dist.values())
    assert sum(dist.values()) == 1.0 or abs(sum(dist.values()) - 1.0) < 1e-9


# ---- APhysDegTransform end-to-end ----

def test_a_physdeg_transform_output_shape_matches_static():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(0.5)
    tfm = APhysDegTransform(BASE_TRANSFORM, ctrl, rng=random.Random(0))
    img = make_synthetic_image(size=(16, 16))
    out = tfm(img)
    assert out.shape == (3, 1, 16, 16)


def test_a_physdeg_transform_chains_stay_within_unlocked_severities():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(0.0)  # only severity 0 unlocked
    tfm = APhysDegTransform(BASE_TRANSFORM, ctrl, rng=random.Random(0))
    for _ in range(30):
        chain = tfm._sample_chain()
        assert all(s == 0 for _, s in chain)


def test_weighted_sample_chain_is_deterministic_given_same_rng_seed():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(1.0)
    dist = ctrl.cell_distribution()
    c1 = weighted_sample_chain(random.Random(42), dist, (2, 3), ID_TYPES)
    c2 = weighted_sample_chain(random.Random(42), dist, (2, 3), ID_TYPES)
    assert c1 == c2


def test_weighted_sample_chain_has_no_duplicate_types():
    ctrl = AdaptivePhysDegController()
    ctrl.set_progress(1.0)
    dist = ctrl.cell_distribution()
    rng = random.Random(7)
    for _ in range(30):
        chain = weighted_sample_chain(rng, dist, (2, 3), ID_TYPES)
        types = [t for t, _ in chain]
        assert len(types) == len(set(types))
