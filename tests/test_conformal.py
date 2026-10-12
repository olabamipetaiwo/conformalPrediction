"""Unit tests for split conformal prediction and coverage metrics (pure logic)."""

from __future__ import annotations

import numpy as np
import pytest

from cap6545.conformal import (
    SplitConformal,
    absolute_residual_scores,
    conformal_quantile,
)
from cap6545.evaluation import coverage_report, empirical_coverage, mean_width


def test_conformal_quantile_finite_sample_rank() -> None:
    scores = np.arange(1, 11, dtype=float)  # 1..10, n=10
    # rank = ceil((10+1)*(1-0.1)) = ceil(9.9) = 10 -> 10th smallest = 10
    assert conformal_quantile(scores, 0.1) == 10.0
    # rank = ceil((10+1)*(1-0.2)) = ceil(8.8) = 9 -> 9th smallest = 9
    assert conformal_quantile(scores, 0.2) == 9.0


def test_conformal_quantile_infinite_when_too_few_points() -> None:
    # n=5, alpha=0.1: rank = ceil(6*0.9) = 6 > 5 -> inf band
    assert conformal_quantile(np.arange(5, dtype=float), 0.1) == float("inf")


def test_conformal_quantile_validates_inputs() -> None:
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)


def test_absolute_residual_scores_plain_and_normalized() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 1.0, 5.0])
    np.testing.assert_allclose(absolute_residual_scores(y, p), [0.5, 1.0, 2.0])
    scale = np.array([0.5, 2.0, 1.0])
    np.testing.assert_allclose(absolute_residual_scores(y, p, scale), [1.0, 0.5, 2.0])
    with pytest.raises(ValueError):
        absolute_residual_scores(y, p, np.array([1.0, 0.0, 1.0]))


def test_split_conformal_intervals_symmetric_and_width() -> None:
    sc = SplitConformal(q=1.5, alpha=0.1)
    pred = np.array([0.0, 10.0])
    lo, hi = sc.intervals(pred)
    np.testing.assert_allclose(lo, [-1.5, 8.5])
    np.testing.assert_allclose(hi, [1.5, 11.5])
    assert mean_width(lo, hi) == pytest.approx(3.0)


def test_normalized_intervals_scale_with_uncertainty() -> None:
    sc = SplitConformal(q=2.0, alpha=0.1)
    pred = np.array([0.0, 0.0])
    scale = np.array([1.0, 3.0])
    lo, hi = sc.intervals(pred, scale=scale)
    # half-widths are q*scale = [2, 6]
    np.testing.assert_allclose(hi - lo, [4.0, 12.0])


def test_empirical_coverage_and_report() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])  # covers 0 and 1, not 2 or 3
    assert empirical_coverage(y, lo, hi) == 0.5
    rep = coverage_report(y, lo, hi, alpha=0.1)
    assert rep.n == 4
    assert rep.target_coverage == pytest.approx(0.9)
    assert rep.empirical_coverage == 0.5
    assert rep.mean_width == pytest.approx(2.0)


def test_split_conformal_marginal_coverage_guarantee() -> None:
    """Averaged over many exchangeable draws, coverage should sit at ~1-alpha.

    Theory: 1-alpha <= E[coverage] <= 1-alpha + 1/(n_cal+1).
    """
    rng = np.random.default_rng(0)
    alpha = 0.1
    n_cal = n_test = 500
    covs = []
    for _ in range(200):
        # exchangeable: predictions are truth plus iid noise, same dist cal/test
        cal_err = rng.standard_normal(n_cal)
        test_err = rng.standard_normal(n_test)
        sc = SplitConformal.calibrate(
            y_cal=cal_err, pred_cal=np.zeros(n_cal), alpha=alpha
        )
        lo, hi = sc.intervals(np.zeros(n_test))
        covs.append(empirical_coverage(test_err, lo, hi))
    mean_cov = float(np.mean(covs))
    # E[coverage] >= 1-alpha in theory; allow small Monte-Carlo slack on the estimate.
    assert 1 - alpha - 0.01 <= mean_cov <= 1 - alpha + 1 / (n_cal + 1) + 0.01
