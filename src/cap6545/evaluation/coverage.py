"""Coverage and interval-width metrics for conformal prediction.

The two quantities every conformal experiment reports: did the intervals cover
the truth at the target rate (coverage), and how wide were they (efficiency).
``coverage_report`` bundles both for a split so Q1/Q3 tables are one call away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ArrayLike = np.ndarray | list[float]


def empirical_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of targets falling within ``[lower, upper]`` (inclusive)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lower: ArrayLike, upper: ArrayLike) -> float:
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def median_width(lower: ArrayLike, upper: ArrayLike) -> float:
    return float(np.median(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


@dataclass(frozen=True)
class CoverageReport:
    n: int
    alpha: float
    target_coverage: float
    empirical_coverage: float
    mean_width: float
    median_width: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def coverage_report(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> CoverageReport:
    """Coverage + width summary for one set of intervals at miscoverage ``alpha``."""
    return CoverageReport(
        n=int(np.asarray(y_true).size),
        alpha=alpha,
        target_coverage=1.0 - alpha,
        empirical_coverage=empirical_coverage(y_true, lower, upper),
        mean_width=mean_width(lower, upper),
        median_width=median_width(lower, upper),
    )
