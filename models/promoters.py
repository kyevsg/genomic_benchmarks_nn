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
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_neurons)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_neurons, out_features=num_classes)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, sequences):
        """
        This function is the forward pass of the neural network.
        
        Inputs:
            sequences: Input genomic sequences
            
        Returns:
            Predicted probabilities
        """
        try:
            if isinstance(sequences, torch.Tensor):  # tensor input from DataLoader
                if len(sequences.shape) == 3:  # shape is [batch_size, seq_length, 4]
                    batch_size = sequences.shape[0]
                    batch = sequences.reshape(batch_size, -1)
                else:  # shape is [batch_size, seq_length]
                    batch_size = sequences.shape[0]
                    seq_length = sequences.shape[1]
                    batch = torch.zeros((batch_size, seq_length * 4))
                    for i in range(batch_size):
                        for j, val in enumerate(sequences[i]):
                            batch[i, j * 4 + val] = 1
            else:  # string input
                nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                batch_size = len(sequences)
                seq_length = len(sequences[0])
                batch = torch.zeros((batch_size, seq_length * 4))
                
                for i, seq in enumerate(sequences):
                    for j, nuc in enumerate(seq):
                        if nuc in nuc_to_idx:
                            batch[i, j * 4 + nuc_to_idx[nuc]] = 1

            hidden = self.activation(self.linear1(batch))
            output = self.output_activation(self.linear2(hidden))
            return output
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

def train_model(model, dataset_train, learning_rate=0.001, num_epochs=2):
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
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(num_epochs):
            for sequences, labels in train_loader:
                optimizer.zero_grad()
                labels = labels.float().view(-1, 1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
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
        model.eval()
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs = model(sequences)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()
        
        return correct / total
    
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

        # sample data for shape verification (debug)
        sample_seq, sample_label = dataset_train[0]
        sequence_length = len(sample_seq)
        
        logger.info(f"Sample sequence shape: {len(sample_seq)}")
        logger.info(f"Sample sequence: {sample_seq[:10]}...")
        
        # initializing model
        model = PromoterClassifier(sequence_length, hidden_neurons, num_classes)
        
        # model architecture (debug)
        logger.info(f"Model architecture:")
        logger.info(f"Input size: {sequence_length * 4}")
        logger.info(f"Hidden neurons: {hidden_neurons}")
        logger.info(f"Output classes: {num_classes}")
        
        # training and evaluating
        trained_model, accuracy = train_model(model, dataset_train, learning_rate, num_epochs)
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
