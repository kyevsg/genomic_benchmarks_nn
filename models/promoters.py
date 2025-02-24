from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters
import torch
import torch.nn as nn
from typing import Tuple
import logging


# configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromoterClassifier(nn.Module):
    def __init__(self, sequence_length: int, hidden_neurons: int, num_classes: int):
        """
        This function initializes the promoter classifier.
        
        Inputs:
            sequence_length: Length of the input genomic sequence
            hidden_neurons: Number of neurons in hidden layer
            num_classes: Number of output classes (1 for binary classification)
        """
        super(PromoterClassifier, self).__init__()
        input_size = sequence_length * 4
        self.linear1 = nn.Linear(input_size, hidden_neurons)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_neurons, num_classes)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        This function is the forward pass of the neural network.
        
        Inputs:
            sequences: Input genomic sequences
            
        Returns:
            Predicted probabilities
        """
        try:   # converting string sequences to one-hot encoding if needed
            if isinstance(sequences[0], str):
                nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                batch_size = len(sequences)
                seq_length = len(sequences[0])
                encoded = torch.zeros((batch_size, seq_length, 4))
                
                for i, seq in enumerate(sequences):
                    for j, nuc in enumerate(seq):
                        encoded[i, j, nuc_to_idx[nuc]] = 1
                
                # reshaping tensor and flattening the one-hot encoded sequences
                batch = encoded.view(batch_size, seq_length * 4)
                
                # debugging (get rid of later)
                logger.debug(f"Encoded shape: {encoded.shape}")
                logger.debug(f"Batch shape after reshape: {batch.shape}")
                logger.debug(f"Linear1 weight shape: {self.linear1.weight.shape}")
            else:
                batch = sequences.view(sequences.size(0), -1)
            
            hidden = self.activation(self.linear1(batch))
            output = self.output_activation(self.linear2(hidden))
            return output
        except RuntimeError as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

def train_model(model: PromoterClassifier, 
                dataset_train: torch.utils.data.Dataset,
                learning_rate: float,
                num_epochs: int) -> Tuple[PromoterClassifier, float]:
    """
    This function trains the promoter classifier model.
    
    Inputs:
        model: The neural network model
        dataset: Training dataset
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        
    Returns:
        The trained model and final accuracy
    """
    try:
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            for sequences, labels in dataset_train:
                optimizer.zero_grad()
                labels = float(labels)
                predictions = model(sequences)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataset_train)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # evaluating the model
        accuracy = evaluate_model(model, dataset_train)
        return model, accuracy
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def evaluate_model(model: PromoterClassifier, 
                  dataset_test: torch.utils.data.Dataset) -> float:
    """
    This function evaluates the model's accuracy.
    
    Inputs:
        model: Trained model
        dataset: Evaluation dataset
        
    Returns:
        Classification accuracy
    """
    try:
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in dataset_test:
                predictions = model(sequences)
                predicted_labels = torch.round(predictions)
                total_samples += labels.size(0)
                correct_predictions += (predicted_labels == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        return accuracy
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def main():
    try:
        # params
        hidden_neurons = 64
        num_classes = 1
        learning_rate = 0.001
        num_epochs = 10
        
        # loading dataset
        dataset_train = HumanNontataPromoters(split='train')
        dataset_test = HumanNontataPromoters(split='test')

        sequence_length = len(dataset_test[0][0])
        
        # initializing model
        model = PromoterClassifier(sequence_length, hidden_neurons, num_classes)

        # debugging (get rid of later)
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Input size to first linear layer: {sequence_length * 4}")
        
        # training and evaluating
        trained_model, accuracy = train_model(model, dataset_train, learning_rate, num_epochs)
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
