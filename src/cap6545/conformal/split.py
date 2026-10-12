"""Split conformal prediction with (optionally normalized) absolute-residual scores.

Split conformal gives finite-sample *marginal* coverage under exchangeability:
score a held-out calibration set, take the conformal quantile of those scores,
then band every test prediction by it. With unit scale this is the standard
absolute-residual method; pass a positive per-point ``scale`` (e.g. an ensemble
or MC-dropout uncertainty) to get the uncertainty-normalized variant, whose
bands widen where the model is less certain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray | list[float]


def absolute_residual_scores(
    y_true: ArrayLike, y_pred: ArrayLike, scale: ArrayLike | None = None
) -> np.ndarray:
    """Nonconformity scores ``|y - yhat|``, divided by ``scale`` when given."""
    resid = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if scale is None:
        return resid
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    return resid / scale


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """The split-conformal quantile of ``scores`` for miscoverage ``alpha``.

    Uses the finite-sample ``(n + 1)`` correction: the band is the k-th smallest
    score with ``k = ceil((n + 1)(1 - alpha))``. If that rank exceeds ``n`` (too
    few calibration points for the requested coverage), returns ``inf`` so the
    band trivially covers everything, matching the theory.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass(frozen=True)
class SplitConformal:
    """A calibrated split-conformal band: half-width ``q`` on the score scale."""

    q: float
    alpha: float

    @classmethod
    def calibrate(
        cls,
        y_cal: ArrayLike,
        pred_cal: ArrayLike,
        alpha: float = 0.1,
        *,
        scale_cal: ArrayLike | None = None,
    ) -> SplitConformal:
        """Fit the band on a calibration set's residuals (optionally normalized)."""
        scores = absolute_residual_scores(y_cal, pred_cal, scale_cal)
        return cls(q=conformal_quantile(scores, alpha), alpha=alpha)

    def intervals(
        self, pred: ArrayLike, *, scale: ArrayLike | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds for each prediction; pass ``scale`` iff calibrated with one."""
        pred = np.asarray(pred, dtype=float)
        half = self.q if scale is None else self.q * np.asarray(scale, dtype=float)
        return pred - half, pred + half
