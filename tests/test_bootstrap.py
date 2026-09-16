import random

import pytest

from eval.bootstrap import group_bootstrap_ci
from eval.metrics import macro_f1
from eval.records import PredictionRecord


def _rec(sample_id, group_id, true, pred, conf=0.9):
    return PredictionRecord(sample_id, group_id, true, pred, conf)


def _macro_f1_metric(records, num_classes=2):
    if not records:
        return 0.0
    preds = [r.pred_label for r in records]
    labels = [r.true_label for r in records]
    return macro_f1(preds, labels, num_classes)


def test_point_estimate_equals_metric_on_full_unresampled_data():
    records = [_rec(f"s{i}", f"g{i}", i % 2, i % 2) for i in range(10)]  # all correct
    ci = group_bootstrap_ci(records, _macro_f1_metric, n_boot=200, rng=random.Random(0))
    assert ci.point == pytest.approx(1.0)


def test_ci_bounds_bracket_the_point_estimate():
    rng = random.Random(0)
    records = []
    for i in range(40):
        true = i % 2
        pred = true if rng.random() < 0.7 else 1 - true
        records.append(_rec(f"s{i}", f"g{i}", true, pred))
    ci = group_bootstrap_ci(records, _macro_f1_metric, n_boot=500, rng=random.Random(1))
    assert ci.lower <= ci.point <= ci.upper


def test_reproducible_given_the_same_rng_seed():
    records = [_rec(f"s{i}", f"g{i // 2}", i % 2, i % 2) for i in range(20)]
    ci1 = group_bootstrap_ci(records, _macro_f1_metric, n_boot=100, rng=random.Random(42))
    ci2 = group_bootstrap_ci(records, _macro_f1_metric, n_boot=100, rng=random.Random(42))
    assert ci1 == ci2


def test_empty_records_raises():
    with pytest.raises(ValueError):
        group_bootstrap_ci([], _macro_f1_metric)


def test_invalid_alpha_raises():
    records = [_rec("s0", "g0", 0, 0)]
    with pytest.raises(ValueError):
        group_bootstrap_ci(records, _macro_f1_metric, alpha=1.5)


def test_resampling_is_group_level_not_record_level():
    """10 groups x 5 records each, all identical within a group. A metric that
    counts distinct groups present in a draw must show real duplicate-draw
    variance (mean well below 10) if groups — not individual records — are the
    resampling unit; if the code accidentally resampled records, this exact
    all-identical-within-group setup wouldn't distinguish the two, so the sharp
    check is against the known closed-form expectation for group-level
    resampling with replacement: E[distinct] = n*(1-(1-1/n)^n) ~= 6.51 for n=10."""
    records = []
    for g in range(10):
        for j in range(5):
            records.append(_rec(f"s{g}_{j}", f"g{g}", 0, 0))

    def distinct_group_count(recs):
        return float(len({r.group_id for r in recs}))

    ci = group_bootstrap_ci(records, distinct_group_count, n_boot=3000, rng=random.Random(0))
    assert ci.point == 10.0
    mean_distinct = sum(ci.draws) / len(ci.draws)
    assert 5.5 < mean_distinct < 7.5  # close to the 6.51 closed-form expectation
    assert ci.upper < 10.0  # getting all 10 back is extremely unlikely, not the norm
