
import numpy as np
import torch
from torch.utils.data import Dataset

aa_to_int = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 
    'X': 20  # For padding
}

ss_to_int = {
    'h': 0,
    'e': 1,
    '_': 2,
    'X': 3  # For padding
}

def encode_and_pad_sequence(seq, encoding_dict, max_length):
    # Encode sequence
    encoded_seq = [encoding_dict.get(aa, encoding_dict['X']) for aa in seq]
    # Pad sequence
    padded_seq = encoded_seq[:max_length] + [encoding_dict['X']] * max(0, max_length - len(seq))
    return padded_seq

def preprocess_data(file_path, aa_to_int, ss_to_int, max_length):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sequences, structures = [], []
    temp_seq, temp_struc = '', ''

    for line in lines:
        if line.startswith("#") or line.strip() == '':
            continue
        elif line.strip() == '<>' or line.strip() == '<end>':
            if temp_seq == '' or temp_struc == '':
                continue
            sequences.append(temp_seq)
            structures.append(temp_struc)
            temp_seq, temp_struc = '', ''
        else:
            temp_seq += line[0]
            temp_struc += line[2]
        
    encoded_sequences = [encode_and_pad_sequence(seq, aa_to_int, max_length) for seq in sequences]
    encoded_structures = [encode_and_pad_sequence(ss, ss_to_int, max_length) for ss in structures]

    return np.array(encoded_sequences), np.array(encoded_structures)

class ProteinDataset(Dataset):
    def __init__(self, sequences, structures):
        """
        Args:
            sequences (numpy.ndarray): Encoded and padded amino acid sequences.
            structures (numpy.ndarray): Encoded and padded secondary structure labels.
        """
        self.sequences = sequences
        self.structures = structures

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        structure = torch.tensor(self.structures[idx], dtype=torch.long)
        return sequence, structure
    
