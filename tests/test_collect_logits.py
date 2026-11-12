import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skin_lesion_uncertainty.evaluate import collect_logits


def test_collect_logits_is_deterministic_with_dropout():
    """eval() must switch dropout off, so two passes give identical logits."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 3))
    data = TensorDataset(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
    loader = DataLoader(data, batch_size=4)
    first, labels = collect_logits(model, loader, "cpu")
    second, _ = collect_logits(model, loader, "cpu")
    assert first.shape == (10, 3)
    assert labels.shape == (10,)
    np.testing.assert_array_equal(first, second)
