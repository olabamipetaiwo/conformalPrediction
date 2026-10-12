"""Evaluation metrics for conformal prediction: coverage and interval width."""

from cap6545.evaluation.coverage import (
    CoverageReport,
    coverage_report,
    empirical_coverage,
    mean_width,
    median_width,
)

__all__ = [
    "CoverageReport",
    "coverage_report",
    "empirical_coverage",
    "mean_width",
    "median_width",
]
