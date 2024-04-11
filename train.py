from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from data_processing import preprocess_data, aa_to_int, ss_to_int, ProteinDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from model import RNN

# Load and preprocess training data
max_length = 498 # Max length of sequences in the dataset
train_sequences, train_structures = preprocess_data("protein-secondary-structure.train", aa_to_int, ss_to_int, max_length)
test_sequences, test_structures = preprocess_data("protein-secondary-structure.test", aa_to_int, ss_to_int, max_length)

# Create DataLoader objects
train_dataset = ProteinDataset(train_sequences, train_structures)
test_dataset = ProteinDataset(test_sequences, test_structures)

start_time = time.time()

# Hyperparameters
vocab_size = len(aa_to_int)  # Number of unique amino acids + padding
hidden_size = 128
num_classes = len(ss_to_int)  # Number of unique secondary structure labels + padding
num_epochs = 30
batch_size = 4
learning_rate = 0.0001
embedding_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss, and optimizer
model = RNN(vocab_size, embedding_dim, hidden_size, num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
total_step = len(train_loader)

def train_model():
    model.train()
    for epoch in range(num_epochs):
        for i, (sequences, structures) in enumerate(train_loader):
            sequences = sequences.long()
            structures = structures.long()

            sequences = sequences.to(device)
            structures = structures.to(device)

            # Forward pass
            outputs = model(sequences)  
            outputs = outputs.view(-1, num_classes)
            structures = structures.view(-1) 
            mask = (structures != 3)
            outputs_masked = outputs[mask]
            structures_masked = structures[mask]

            # loss = criterion(outputs, structures)
            loss = criterion(outputs_masked, structures_masked)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0: # Print every 100 batches
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def test_model():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        mask_correct = 0
        mask_total = 0
        temp_best_accuracy = 0

        for sequences, structures in test_loader:
            sequences = sequences.long()
            structures = structures.long()

            sequences = sequences.to(device)
            structures = structures.to(device)

            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 2)

            # Calculate total accuracy including padding
            total += structures.size(0) * structures.size(1)
            correct += (predicted == structures).sum().item()

            # Create a mask to exclude padding tokens (assuming padding token is encoded as 3)
            mask = (structures != 3)
            masked_predicted = predicted[mask]
            masked_structures = structures[mask]

            # Calculate accuracy excluding padding
            mask_total += mask.sum().item()
            mask_correct += (masked_predicted == masked_structures).sum().item()
            

        print('Test Accuracy of the model on the test sequences (excluding padding): {} %'.format(100 * mask_correct / mask_total))

train_model()
# torch.save(model.state_dict(), 'protein_model.pth')
test_model()

print("--- %s seconds ---" % (time.time() - start_time))
