import numpy as np
from scipy.special import softmax

from skin_lesion_uncertainty.conformal import (
    conformal_scores,
    conformal_threshold,
    coverage_and_size,
    prediction_sets,
)


def test_split_conformal_coverage_on_exchangeable_data():
    """Average coverage over many calibration draws is 1 - alpha (theory: 0.900 to 0.905)."""
    rng = np.random.default_rng(0)
    alpha, n_cal, n_test, n_classes = 0.1, 200, 1000, 5
    coverages = []
    for _ in range(200):
        probs = softmax(2 * rng.normal(size=(n_cal + n_test, n_classes)), axis=1)
        # One label per row drawn from probs, so the data are exchangeable by construction.
        labels = (rng.random((len(probs), 1)) > probs.cumsum(axis=1)).sum(axis=1)
        labels = np.minimum(labels, n_classes - 1)
        q = conformal_threshold(conformal_scores(probs[:n_cal], labels[:n_cal]), alpha)
        sets = prediction_sets(probs[n_cal:], q)
        coverages.append(coverage_and_size(sets, labels[n_cal:])[0])
    assert 0.89 <= np.mean(coverages) <= 0.92
