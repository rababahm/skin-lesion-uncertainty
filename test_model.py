import torch
from skin_lesion_uncertainty.model import SmallCNN


def test_output_shape():
    model = SmallCNN(n_classes=7)
    x = torch.zeros(8, 3, 28, 28)
    assert model(x).shape == (8, 7)


def test_parameter_count():
    # Pins the architecture: an accidental change to a layer fails this test.
    model = SmallCNN(n_classes=7)
    assert sum(p.numel() for p in model.parameters()) == 41_351
