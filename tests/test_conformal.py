import numpy as np
import pytest

from skin_lesion_uncertainty.conformal import (
    classwise_prediction_sets,
    classwise_thresholds,
    conformal_scores,
    conformal_threshold,
    coverage_and_size,
    partial_classwise_thresholds,
    prediction_sets,
)


def test_conformal_scores():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 1])
    np.testing.assert_allclose(conformal_scores(probs, labels), [0.3, 0.2])


def test_threshold_nine_scores():
    scores = np.arange(1, 10) / 10  # n = 9: rank ceil(10 * 0.9) = 9, the largest score
    assert conformal_threshold(scores, alpha=0.1) == pytest.approx(0.9)


def test_threshold_eight_scores():
    scores = np.arange(1, 9) / 10  # n = 8: rank 9 > 8, so no finite threshold
    assert np.isinf(conformal_threshold(scores, alpha=0.1))


def test_prediction_sets_includes_boundary():
    probs = np.array([[0.75, 0.25], [0.5, 0.5]])
    expected = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(prediction_sets(probs, 0.5), expected)


def test_infinite_threshold_includes_all_classes():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert prediction_sets(probs, float("inf")).all()


def test_coverage_and_size():
    sets = np.array([[True, False], [True, True], [False, True]])
    labels = np.array([0, 1, 0])
    coverage, size = coverage_and_size(sets, labels)
    assert coverage == pytest.approx(2 / 3)
    assert size == pytest.approx(4 / 3)


def test_classwise_thresholds():
    scores = np.r_[np.arange(1, 10) / 10, np.arange(1, 9) / 10]  # 9 of class 0, 8 of class 1
    labels = np.r_[np.zeros(9, dtype=int), np.ones(8, dtype=int)]
    thresholds = classwise_thresholds(scores, labels, n_classes=2, alpha=0.1)
    assert thresholds[0] == pytest.approx(0.9)
    assert np.isinf(thresholds[1])  # 8 images are too few for 90%


def test_classwise_prediction_sets():
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    thresholds = np.array([0.5, 0.2])  # class 0 needs p >= 0.5, class 1 needs p >= 0.8
    expected = np.array([[True, False], [False, False]])  # Mondrian sets can be empty
    np.testing.assert_array_equal(classwise_prediction_sets(probs, thresholds), expected)


def test_partial_classwise_thresholds():
    rng = np.random.default_rng(0)
    scores, labels = rng.random(200), rng.integers(0, 3, 200)
    thresholds = partial_classwise_thresholds(scores, labels, 3, 0.1, classes=(2,))
    assert thresholds[0] == thresholds[1] == conformal_threshold(scores, 0.1)
    assert thresholds[2] == conformal_threshold(scores[labels == 2], 0.1)
