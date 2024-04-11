import torch
import sys
from model import RNN
from data_processing import encode_and_pad_sequence, aa_to_int

vocab_size = 21
embedding_dim = 100
hidden_size = 128
num_classes = 4
max_length = 498

model = RNN(vocab_size, embedding_dim, hidden_size, num_classes)
model.load_state_dict(torch.load('protein_model.pth'))
model.eval()

int_to_ss = {0: 'h', 1: 'e', 2: '_', 3: 'X'}

def decode_predictions(predictions):
    return ''.join(int_to_ss[int(pred)] for pred in predictions if pred != 3)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <amino_acid_sequence>")
        return
    
    seq = sys.argv[1]
    if len(seq) > max_length:
        print(f"Sequence length should not exceed {max_length} characters.")
        return

    encoded_seq = encode_and_pad_sequence(seq, aa_to_int, max_length)
    input_tensor = torch.tensor([encoded_seq], dtype=torch.long)

    with torch.no_grad():
        predictions = model(input_tensor)
        predictions = predictions.argmax(dim=2).squeeze(0).numpy()

    predictions = predictions[:len(seq)]
    decoded_predictions = decode_predictions(predictions)
    print("Predicted Secondary Structure:", decoded_predictions)

if __name__ == "__main__":
    main()
