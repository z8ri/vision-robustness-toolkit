import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from degradation.params import ID_TYPES, N_SEVERITIES, OOD_FAMILIES, OOD_TYPES
from eval.protocol import CorruptionDataset, run_id_protocol, run_ood_protocol
from tests._synthetic import make_synthetic_samples

TRANSFORM = T.Compose([T.Resize((32, 32)), T.PILToTensor(), lambda t: t.float() / 255.0])


def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 2) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s)


def trivial_brightness_threshold_eval(loader: DataLoader) -> float:
    """Stub evaluate_fn: a 'model' that just thresholds mean pixel brightness.
    Real components (S-WFA etc.) will later supply a real checkpoint's eval loop
    with this exact same signature (DataLoader -> macro_f1), so this test proves
    the protocol's wiring/aggregation is correct independent of what model is
    plugged in."""
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        means = imgs.mean(dim=(1, 2, 3))
        all_preds.append((means > 0.4).long())
        all_labels.append(labels)
    return _macro_f1(torch.cat(all_preds), torch.cat(all_labels))


def test_id_protocol_shape_and_aggregation():
    samples = make_synthetic_samples(n_per_class=3, n_classes=2)
    result = run_id_protocol(samples, TRANSFORM, trivial_brightness_threshold_eval, batch_size=4)

    assert set(result.per_type.keys()) == set(ID_TYPES)
    for deg_type, s_results in result.per_type.items():
        assert set(s_results.keys()) == {"s1", "s2", "s3", "mean"}
        expected_mean = round(sum(s_results[f"s{i + 1}"] for i in range(N_SEVERITIES)) / N_SEVERITIES, 4)
        assert s_results["mean"] == expected_mean

    all_cells = [v for s_results in result.per_type.values()
                 for k, v in s_results.items() if k != "mean"]
    assert len(all_cells) == len(ID_TYPES) * N_SEVERITIES  # 7 x 3 = 21
    assert result.mPC == round(sum(all_cells) / len(all_cells), 4)
    assert result.family_means == {}  # no family grouping defined for the ID regime


def test_ood_protocol_shape_and_family_means():
    samples = make_synthetic_samples(n_per_class=3, n_classes=2)
    result = run_ood_protocol(samples, TRANSFORM, trivial_brightness_threshold_eval, batch_size=4)

    assert set(result.per_type.keys()) == set(OOD_TYPES)
    all_cells = [v for s_results in result.per_type.values()
                 for k, v in s_results.items() if k != "mean"]
    assert len(all_cells) == len(OOD_TYPES) * N_SEVERITIES  # 8 x 3 = 24

    assert set(result.family_means.keys()) == set(OOD_FAMILIES.keys())
    # "digital" family has exactly one member (pixelate) -> family mean == its own mean.
    assert result.family_means["digital"] == result.per_type["pixelate"]["mean"]


def test_protocol_is_reproducible_across_runs():
    samples = make_synthetic_samples(n_per_class=3, n_classes=2)
    r1 = run_id_protocol(samples, TRANSFORM, trivial_brightness_threshold_eval, batch_size=4)
    r2 = run_id_protocol(samples, TRANSFORM, trivial_brightness_threshold_eval, batch_size=4)
    assert r1.per_type == r2.per_type
    assert r1.mPC == r2.mPC


def test_corruption_dataset_rejects_unknown_regime():
    import pytest
    samples = make_synthetic_samples(n_per_class=1, n_classes=2)
    with pytest.raises(ValueError):
        CorruptionDataset(samples, TRANSFORM, "motion_blur", 0, regime="not_id_or_ood")
