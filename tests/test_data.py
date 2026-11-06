import numpy as np
import pytest

from skin_lesion_uncertainty.data import SPLITS, load_split, load_splits

EXPECTED_SIZES = {"train": 7007, "val": 1003, "test": 2005}


@pytest.fixture(scope="module")
def splits():
    # Load once for every test in this file: reading the data is the slow part.
    return load_splits(size=28)


def test_all_splits_present(splits):
    assert set(splits) == set(SPLITS)


@pytest.mark.parametrize("split", SPLITS)
def test_split_shapes_and_types(splits, split):
    X, y = splits[split]
    n = EXPECTED_SIZES[split]
    assert X.shape == (n, 28, 28, 3)
    assert X.dtype == np.uint8
    assert y.shape == (n,)
    assert y.dtype == np.int64


@pytest.mark.parametrize("split", SPLITS)
def test_labels_in_range(splits, split):
    _, y = splits[split]
    assert y.min() >= 0
    assert y.max() <= 6


def test_train_has_all_seven_classes(splits):
    _, y = splits["train"]
    assert len(np.unique(y)) == 7


@pytest.mark.parametrize("split", ["training", "validation", "testing", "", "TRAIN"])
def test_invalid_split_raises(split):
    with pytest.raises(ValueError):
        load_split(split)
