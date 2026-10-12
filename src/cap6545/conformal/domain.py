"""Domain-classifier density ratios for weighted conformal.

The likelihood ratio ``w(x) = p_test(x) / p_cal(x)`` is estimated the standard way
(probabilistic classification trick): train a classifier to tell calibration inputs
(label 0) from test inputs (label 1). For its probability ``c(x) = P(test | x)``,

    p_test(x) / p_cal(x) = (c / (1 - c)) * (n_cal / n_test),

where the ``n_cal / n_test`` factor removes the sampling-prior bias so the ratio is 1
when the two input distributions coincide (no shift). A near-0.5 classifier AUC means
no detectable shift and weights collapse to 1, recovering plain split conformal.

Probabilities are cross-fitted (out-of-fold) so a calibration point's own weight is
not computed from a model that memorized it, and clipped away from 0/1 to keep ratios
finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ArrayLike = np.ndarray


@dataclass(frozen=True)
class DomainRatios:
    """Estimated density ratios ``w(x)=p_test/p_cal`` for calibration and test points."""

    cal_weights: np.ndarray
    test_weights: np.ndarray
    auc: float  # cross-validated discrimination; ~0.5 means no detectable shift


def _make_classifier(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000),
    )


def domain_density_ratios(
    x_cal: ArrayLike,
    x_test: ArrayLike,
    *,
    clip: float = 1e-3,
    C: float = 1.0,
    folds: int = 5,
    seed: int = 0,
) -> DomainRatios:
    """Density ratios for calibration/test inputs via a cross-fitted domain classifier.

    ``clip`` bounds ``P(test|x)`` into ``[clip, 1-clip]`` so ratios stay finite;
    ``C`` is the logistic-regression inverse regularization strength.
    """
    xc = np.asarray(x_cal, dtype=float)
    xt = np.asarray(x_test, dtype=float)
    n_cal, n_test = xc.shape[0], xt.shape[0]

    x = np.vstack([xc, xt])
    y = np.concatenate([np.zeros(n_cal, dtype=int), np.ones(n_test, dtype=int)])

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(_make_classifier(C), x, y, cv=cv, method="predict_proba")[:, 1]
    proba = np.clip(proba, clip, 1.0 - clip)

    prior = n_cal / n_test  # undo the sampling base rates
    ratios = (proba / (1.0 - proba)) * prior

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, proba))
    return DomainRatios(cal_weights=ratios[:n_cal], test_weights=ratios[n_cal:], auc=auc)
