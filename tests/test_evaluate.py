import numpy as np
import pytest
from scipy.special import softmax

from skin_lesion_uncertainty.evaluate import expected_calibration_error, fit_temperature, top_label


def test_ece_known_value_when_always_certain():
    confidence = np.ones(1000)
    correct = np.r_[np.ones(800), np.zeros(200)]  # 80% right at confidence 1.0
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.2)


def test_ece_small_for_calibrated_data():
    rng = np.random.default_rng(0)
    n = 100_000
    confidence = rng.uniform(0.2, 1.0, n)
    correct = rng.random(n) < confidence

    assert expected_calibration_error(confidence, correct) < 0.01


def test_top_label_on_hand_computed_logits():
    logits = np.array([[0.0, np.log(3)], [np.log(3), 0.0]])
    labels = np.array([1, 1])

    confidence, correct = top_label(logits, labels)

    np.testing.assert_allclose(confidence, [0.75, 0.75])
    np.testing.assert_array_equal(correct, [1, 0])


def test_fit_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    z = 2.0 * rng.standard_normal((5000, 7))
    cumulative = softmax(z, axis=1).cumsum(axis=1)
    labels = (rng.random((5000, 1)) > cumulative).sum(axis=1)  # one draw per row from softmax(z)
    # Labels follow softmax(z), so logits 2z are over-confident by exactly a factor of 2.
    assert fit_temperature(2.0 * z, labels) == pytest.approx(2.0, abs=0.1)
