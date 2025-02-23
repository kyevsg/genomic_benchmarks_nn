import pytest
import torch
from models.promoters import PromoterClassifier, train_model, evaluate_model
from genomic_benchmarks.dataset_getters.pytorch_datasets import human_nontata_promoters

@pytest.fixture
def model():
    return PromoterClassifier(sequence_length=100, hidden_neurons=64, num_classes=1)

@pytest.fixture
def test_dataset():
    # need to add testing dataset here and for testing forward pass

def test_model_initialization(model):
    assert isinstance(model, PromoterClassifier)
    assert hasattr(model, 'linear1')
    assert hasattr(model, 'linear2')

def test_forward_pass(model):
    input_sequence = 
    output = model(input_sequence)
    assert output.shape == (1, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_model_training(model, test_dataset):
    trained_model, accuracy = train_model(
        model, 
        test_dataset,
        learning_rate=0.001,
        num_epochs=2
    )
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_model_evaluation(model, test_dataset):
    accuracy = evaluate_model(model, test_dataset)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_invalid_input_shape(model):
    with pytest.raises(RuntimeError):
        invalid_input = torch.randn(1, 50)
        model(invalid_input)
