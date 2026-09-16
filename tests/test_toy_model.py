import torch

from deploy.toy_model import ToyDefectClassifier


def test_forward_shape():
    model = ToyDefectClassifier(num_classes=5)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)


def test_forward_works_for_a_range_of_batch_sizes():
    model = ToyDefectClassifier(num_classes=3)
    for batch in (1, 2, 8):
        x = torch.randn(batch, 3, 32, 32)
        assert model(x).shape == (batch, 3)


def test_actually_contains_swfa():
    """Sanity check that this is exercising the real component and not a stand-in
    that quietly skips it."""
    from models.wfa import SWFA
    model = ToyDefectClassifier()
    assert isinstance(model.wfa, SWFA)
