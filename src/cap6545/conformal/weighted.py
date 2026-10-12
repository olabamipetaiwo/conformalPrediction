"""Weighted split conformal for covariate shift (Tibshirani et al. 2019).

Plain split conformal assumes calibration and test points are exchangeable. Under
covariate shift the test inputs are drawn from a different distribution, and the
unweighted conformal quantile loses its coverage guarantee -- the undercoverage
we see calibrating on LP-PDBBind val and evaluating on the shifted test split.

Weighting each calibration score by the likelihood ratio ``w(x) = p_test(x)/p_cal(x)``
restores coverage. For a test point with weight ``w_t`` the band is the level
``1 - alpha`` quantile of the distribution that puts mass ``w(x_i)`` on each
calibration score ``s_i`` and mass ``w_t`` on ``+inf`` (all normalized). The infinite
atom is why an isolated test point -- one the calibration set barely covers -- can
get an infinite (i.e. fully uncertain) band, exactly as the theory prescribes.

The likelihood ratios come from a domain classifier; see ``conformal.domain``.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def weighted_conformal_quantile(
    scores: ArrayLike,
    cal_weights: ArrayLike,
    test_weights: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """Per-test-point weighted conformal quantiles of ``scores``.

    Returns one band half-width per test point (shape ``(m,)``). For test point
    ``j`` the threshold is the smallest calibration score whose cumulative
    normalized weight reaches ``1 - alpha`` on the distribution

        sum_i w_i * delta(s_i)  +  w_t[j] * delta(+inf),

    all divided by ``(sum_i w_i) + w_t[j]``. When that mass is only reached by the
    ``+inf`` atom the band is infinite, matching the weighted-exchangeability result.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(scores, dtype=float)
    wc = np.asarray(cal_weights, dtype=float)
    wt = np.asarray(test_weights, dtype=float)
    if s.size == 0:
        raise ValueError("need at least one calibration score")
    if s.shape != wc.shape:
        raise ValueError("scores and cal_weights must have the same shape")
    if np.any(wc <= 0) or np.any(wt <= 0):
        raise ValueError("weights must be strictly positive")

    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    w_sorted = wc[order]
    cum = np.cumsum(w_sorted)  # cumulative calibration weight, ascending score
    total = cum[-1] + wt  # (m,) per-test normalizer including the +inf atom

    targets = (1.0 - alpha) * total  # (m,)
    # First sorted score whose cumulative weight >= target; == cum.size means the
    # requested mass is only met by the +inf atom -> infinite band.
    idx = np.searchsorted(cum, targets, side="left")
    out = np.full(wt.shape, np.inf, dtype=float)
    inb = idx < s_sorted.size
    out[inb] = s_sorted[idx[inb]]
    return out
