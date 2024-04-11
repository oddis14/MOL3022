from data_processing import aa_to_int, ss_to_int
import torch

int_to_aa = {v: k for k, v in aa_to_int.items()}
int_to_ss = {v: k for k, v in ss_to_int.items()}

def print_int_sequences_to_strings(actual, predicted):
    """
    Convert integer-encoded sequences to strings.
    
    Parameters:
    - actual: A 2D tensor (batch_size, seq_length) of actual sequences.
    - predicted: A 2D tensor (batch_size, seq_length) of predicted sequences.
    
    Returns:
    - actual_strings: A list of strings of actual sequences.
    - predicted_strings: A list of strings of predicted sequences.
    """
    actual_strings = "".join([int_to_ss[actual[i].item()] for i in range(actual.size(0))])
    predicted_strings = "".join([int_to_ss[predicted[i].item()] for i in range(predicted.size(0))])
    print(actual_strings)
    print(predicted_strings)


def calculate_accuracy(actual, predicted, padding_index=3):
    """
    Calculate the accuracy, excluding the padding tokens.
    
    Parameters:
    - actual: tensor of actual labels
    - predicted: tensor of predicted labels
    - padding_index: the label index used for padding tokens
    
    Returns:
    - accuracy: float, the calculated accuracy
    """
    mask = (actual != padding_index)  # Create a mask for actual labels not equal to padding_index
    correct_predictions = (actual[mask] == predicted[mask]).sum().item()
    total_predictions = mask.sum().item()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy
