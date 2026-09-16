import pytest
import torch

from models.wavelet import haar_dwt2d, haar_idwt2d


def test_dwt_band_shapes():
    x = torch.randn(2, 6, 16, 24)
    ll, lh, hl, hh = haar_dwt2d(x)
    for band in (ll, lh, hl, hh):
        assert band.shape == (2, 6, 8, 12)


def test_dwt_then_idwt_reconstructs_input():
    """Orthonormal Haar => perfect reconstruction (up to float precision)."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 20, 20)
    recon = haar_idwt2d(*haar_dwt2d(x))
    assert torch.allclose(recon, x, atol=1e-5)


def test_energy_is_conserved_parseval():
    """||x||^2 == ||LL||^2 + ||LH||^2 + ||HL||^2 + ||HH||^2 for an orthonormal
    transform (Table/Eq 9 in the paper draft's frequency-unification section)."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 12, 12)
    ll, lh, hl, hh = haar_dwt2d(x)
    total_energy = x.pow(2).sum()
    band_energy = sum(b.pow(2).sum() for b in (ll, lh, hl, hh))
    assert torch.allclose(total_energy, band_energy, rtol=1e-4)


def test_odd_spatial_size_raises_clear_error():
    x = torch.randn(1, 2, 15, 16)  # odd height
    with pytest.raises(ValueError):
        haar_dwt2d(x)
