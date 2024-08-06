import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

### Defining the CNN model ###

# NOTE All pytorch models inherit from nn.Module
class SentimentCNN(nn.Module):
    # Setting up necessary hyperparameters for this project.
    def __init__(self, vocab_size, embed_dim, num_classes):

        # Calling super class's __init__() method to set up.
        super(SentimentCNN, self).__init__()

        # Setting up embedding layer.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Setting up first convolutional layer
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3)
        # Max pooling for CNN layer 1
        self.pool1 = nn.MaxPool1d(kernel_size=3)

        #Setting up CNN layer 2
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3)
        # Global max pool to 1 dimension for easy transition to Dense Layer.
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # First Dense Layer
        self.fc1 = nn.Linear(64, 64)

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

        # Second Dense Layer
        self.fc2 = nn.Linear(64, num_classes)
        # Sigmoid activation for output layer
        self.sigmoid = nn.Sigmoid()

    
    # Defining forward step function
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Loading encoded data and vocab
data = pd.read_pickle("encoded_data.pkl")
vocab = torch.load("vocab.pth")
print("Data Loaded")

# Preparing data for training
tweets = torch.stack(data["text"].tolist())
labels = torch.tensor(data["target"].tolist())
print("Data prepared")

# Setting Hyperparameters
vocab_size = len(vocab)
embed_dim = 128
num_classes = 1 # Binary classification
learning_rate = 0.001
num_epochs = 10
batch_size = 64
patience = 2 # Number of epochs to wait for improvement
min_delta = 0.001 # Minimum change to qualify as improvement


# Initializing model, loss function and optimizer
model = SentimentCNN(vocab_size, embed_dim, num_classes)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Early stopping variables
best_loss = float("inf")
epochs_no_improve = 0

print("Starting Training Loop!")
# Training Loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_batches = len(tweets) // batch_size
    for i in range(0, len(tweets), batch_size):
        batch_tweets = tweets[i:i+batch_size]
        batch_labels = labels[i:i+batch_size].float()

        # Forward pass
        outputs = model(batch_tweets).squeeze(1)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Print progress
        current_batch = i // batch_size + 1
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{current_batch}/{total_batches}], Loss: {epoch_loss/(current_batch):.4f}", end='')

    epoch_loss /= len(tweets) // batch_size
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Checking for early stopping
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

# Saving the trained model
torch.save(model.state_dict(), "sentiment_cnn.pth")
print("Model saved Succesfully!")