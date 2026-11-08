import numpy as np
import pytest
import torch

from skin_lesion_uncertainty.data import (
    SPLITS,
    class_weights,
    load_split,
    load_splits,
    make_loaders,
    to_tensors,
)

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


def test_to_tensors_layout(splits):
    images, labels = splits["val"]
    x, y = to_tensors(images, labels)
    assert x.shape == (len(images), 3, 28, 28)  # channels first, as PyTorch expects
    assert x.dtype == torch.float32
    assert x.min() >= 0.0
    assert x.max() <= 1.0
    assert y.dtype == torch.int64


def test_make_loaders_batch_shape():
    loaders = make_loaders(batch_size=8)
    x, y = next(iter(loaders["val"]))
    assert x.shape == (8, 3, 28, 28)
    assert y.shape == (8,)


def test_class_weights_equalise_classes():
    labels = np.array([0, 0, 0, 1])  # class 0 is three times as common as class 1
    weights = class_weights(labels, n_classes=2)
    assert torch.allclose(weights, torch.tensor([4 / 6, 4 / 2]))
    # Each class then contributes the same total weight: 3 x 0.667 = 1 x 2.0
    assert torch.isclose(weights[0] * 3, weights[1] * 1)
