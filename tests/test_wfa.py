import torch

from models.wfa import SWFA, WFACore, original_wfa


def test_wfa_core_preserves_shape():
    torch.manual_seed(0)
    core = WFACore(channels=6, reduction=3, hf_kernel=3)
    x = torch.randn(2, 6, 10, 10)
    u = core(x)
    assert u.shape == x.shape


def test_wfa_core_tolerates_channels_smaller_than_reduction():
    """reduction=32 with channels=6 would give hidden=0 without the max(1, ...)
    clamp in WFACore.__init__; this must not crash."""
    core = WFACore(channels=6, reduction=32, hf_kernel=3)
    x = torch.randn(1, 6, 8, 8)
    assert core(x).shape == x.shape


def test_swfa_preserves_shape():
    torch.manual_seed(0)
    swfa = SWFA(channels=8, reduction=2, hf_kernel=3)
    x = torch.randn(3, 8, 12, 12)
    assert swfa(x).shape == x.shape


def test_near_identity_at_init():
    """gamma is LayerScale-initialized to 1e-6, so a freshly constructed SWFA
    must be an approximate identity — this is what lets it be inserted into a
    pretrained backbone without disturbing it (design doc §3.3)."""
    torch.manual_seed(1)
    swfa = SWFA(channels=8, reduction=2, hf_kernel=3)
    x = torch.randn(4, 8, 16, 16)
    y = swfa(x)
    rel_diff = (y - x).norm() / x.norm()
    assert rel_diff < 0.01


def test_force_gate_zero_is_exact_identity():
    """g(x) == 0 must zero out the *entire* residual, not just shrink it —
    design doc: 'g(x)≈0：模块接近无WFA的backbone'."""
    torch.manual_seed(2)
    swfa = SWFA(channels=8, reduction=2, hf_kernel=3)
    x = torch.randn(4, 8, 8, 8)
    y = swfa(x, force_gate=0.0)
    assert torch.equal(y, x)


def test_force_gate_one_equals_manual_gamma_times_core_output():
    torch.manual_seed(3)
    swfa = SWFA(channels=8, reduction=2, hf_kernel=3)
    x = torch.randn(4, 8, 8, 8)
    with torch.no_grad():
        u = swfa.core(x)
    y = swfa(x, force_gate=1.0)
    expected = x + swfa.gamma.view(1, -1, 1, 1) * u
    assert torch.allclose(y, expected, atol=1e-6)


def test_original_wfa_equals_swfa_with_gate_forced_to_one():
    """The load-bearing correctness claim of the whole design: with shared
    WFACore/gamma weights, disabling the gate (original_wfa) and forcing the
    gate to 1.0 on a gated SWFA must be numerically identical — not just
    'conceptually similar'. This is the g(x)->1 endpoint from the design doc."""
    torch.manual_seed(4)
    gated = SWFA(channels=8, reduction=2, hf_kernel=3, use_gate=True)
    baseline = original_wfa(channels=8, reduction=2, hf_kernel=3)
    baseline.core.load_state_dict(gated.core.state_dict())
    baseline.gamma.data.copy_(gated.gamma.data)

    x = torch.randn(4, 8, 8, 8)
    y_gated_forced = gated(x, force_gate=1.0)
    y_baseline = baseline(x)
    assert torch.allclose(y_gated_forced, y_baseline, atol=1e-6)


def test_gate_output_is_a_valid_probability_per_sample():
    torch.manual_seed(5)
    swfa = SWFA(channels=8, reduction=2, hf_kernel=3, use_gate=True)
    x = torch.randn(5, 8, 8, 8)
    y, g = swfa(x, return_gate=True)
    assert y.shape == x.shape
    assert g.shape == (5, 1, 1, 1)
    assert torch.all(g > 0) and torch.all(g < 1)


def test_gate_responds_to_spectral_content_not_constant():
    """Two deliberately different spectral profiles (smooth ramp = low-frequency
    heavy, checkerboard = high-frequency heavy) must not get the same gate
    value — otherwise the 'sample-level selective' claim is vacuous."""
    torch.manual_seed(6)
    swfa = SWFA(channels=4, reduction=2, hf_kernel=3, use_gate=True)
    h = w = 16
    smooth = torch.linspace(0, 1, steps=h).view(1, 1, h, 1).expand(1, 4, h, w).contiguous()
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    checker = ((yy + xx) % 2).float().view(1, 1, h, w).expand(1, 4, h, w).contiguous()
    x = torch.cat([smooth, checker], dim=0)  # (2, 4, h, w)

    _, g = swfa(x, return_gate=True)
    assert not torch.allclose(g[0], g[1])


def test_gate_disabled_ignores_spectral_content():
    """Sanity check on `use_gate=False`: g must be exactly 1 regardless of input,
    i.e. original_wfa() really has no sample-level behavior at all."""
    torch.manual_seed(7)
    baseline = original_wfa(channels=4, reduction=2, hf_kernel=3)
    x = torch.randn(3, 4, 8, 8)
    y, g = baseline(x, return_gate=True)
    assert torch.equal(g, torch.ones_like(g))


def test_wfa_core_param_count_matches_documented_budget():
    """At the canonical placement config (channels=384, reduction=32, hf_kernel=3,
    matching two WFA insertions into ConvNeXt-Tiny Stage 3), WFACore should land
    close to the ~0.75M params/module documented in the vNext result account
    (two modules => ~1.5M, +5.4% of the 27.83M backbone) — a concrete
    cross-check, not just trusting the design doc's arithmetic."""
    core = WFACore(channels=384, reduction=32, hf_kernel=3)
    n_params = sum(p.numel() for p in core.parameters())
    assert 0.70e6 < n_params < 0.80e6


def test_spectral_gate_parameter_count_is_negligible():
    """The gate MLP (4 -> gate_hidden -> 1) must stay tiny relative to WFACore —
    the whole design argument is 'one control scalar', not a new capacity source."""
    swfa = SWFA(channels=384, reduction=32, hf_kernel=3, gate_hidden=8)
    gate_params = sum(p.numel() for p in swfa.gate.parameters())
    core_params = sum(p.numel() for p in swfa.core.parameters())
    assert gate_params < 100
    assert gate_params < 0.001 * core_params
