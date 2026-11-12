import numpy as np
import pytest

from skin_lesion_uncertainty.decision import auto_accept, referral_summary


def make_sets(rows: list[list[int]], n_classes: int = 7) -> np.ndarray:
    sets = np.zeros((len(rows), n_classes), dtype=bool)
    for i, classes in enumerate(rows):
        sets[i, classes] = True
    return sets


def test_auto_accept_only_single_benign_sets():
    sets = make_sets([[2], [4], [2, 5], [5]])  # benign, melanoma, two classes, benign
    np.testing.assert_array_equal(auto_accept(sets), [True, False, False, True])


def test_referral_summary():
    sets = make_sets([[2], [4], [2, 5], [5]])
    labels = np.array([2, 4, 5, 4])  # the last one is a melanoma auto-accepted as nevus
    summary = referral_summary(sets, labels)
    assert summary["referral_rate"] == pytest.approx(0.5)
    assert summary["missed_rate"] == pytest.approx(0.25)
    assert summary["missed_share_of_non_benign"] == pytest.approx(0.5)
