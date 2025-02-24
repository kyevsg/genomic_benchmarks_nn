import pytest
import torch
import torch.nn as nn
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters

import sys
sys.path.append("models")
from promoters import PromoterClassifier, train_model, evaluate_model


@pytest.fixture
def test_dataset():
    return HumanNontataPromoters(split='test', version=0)

@pytest.fixture
def model():
    sequence_length = 251
    hidden_neurons = 64
    num_classes = 1
    return PromoterClassifier(sequence_length, hidden_neurons, num_classes)

def test_model_initialization(model):
    assert isinstance(model, PromoterClassifier)
    assert hasattr(model, 'linear1')
    assert hasattr(model, 'linear2')
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)

def test_forward_pass(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    sequences, _ = next(iter(test_loader))
    output = model(sequences)
    assert output.shape == (1, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_model_training(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    trained_model, accuracy = train_model(
        model, 
        test_dataset,
        learning_rate=0.001,
        num_epochs=2
    )
    assert isinstance(trained_model, PromoterClassifier)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_model_evaluation(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate_model(model, test_dataset)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_invalid_input_shape(model):
    with pytest.raises(RuntimeError):
        invalid_sequence = "A" * 125  # shorter than expected sequence
        model([invalid_sequence])
