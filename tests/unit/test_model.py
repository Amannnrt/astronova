import torch
from src.models.resnet_classifier import AstroClassifier


def test_model_forward_pass():
    model = AstroClassifier(num_classes=4)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 4)


def test_model_trainable_params():
    model = AstroClassifier(num_classes=4)
    assert model.trainable_params > 0
