import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Setting device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Defining the CNN model ###

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

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

# Loading the preprocessed data
data = pd.read_pickle("encoded_data.pkl")
vocab = torch.load("vocab.pth")

# Preparing data for training
tweets = torch.stack(data["text"].tolist()).to(device)
labels = torch.tensor(data["target"].tolist()).to(device)

# Splitting the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=12)

# Setting Hyperparameters
vocab_size = len(vocab)
embed_dim = 128
num_classes = 1  # Binary classification
learning_rate = 0.001
num_epochs = 10
batch_size = 32  # Reduced batch size
accumulation_steps = 4  # Gradient accumulation steps
patience = 2
min_delta = 0.001

# Initializing model, loss function and optimizer
model = SentimentCNN(vocab_size, embed_dim, num_classes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Early stopping variables
best_val_loss = float("inf")
epochs_no_improve = 0

def calculate_accuracy(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy.item()

print("Starting Training Loop!")

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = len(X_train) // batch_size
    optimizer.zero_grad()

    for i in range(0, len(X_train), batch_size):
        batch_tweets = X_train[i:i+batch_size]
        batch_labels = y_train[i:i+batch_size].float()

        # Forward pass
        outputs = model(batch_tweets).squeeze(1)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        loss.backward()
        if (i // batch_size + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate accuracy
        batch_accuracy = calculate_accuracy(outputs, batch_labels)

        epoch_loss += loss.item()
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{i//batch_size + 1}/{total_batches}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}", end="")

    epoch_loss /= total_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze(1)
        val_loss = criterion(val_outputs, y_val.float())
        val_accuracy = calculate_accuracy(val_outputs, y_val)

    print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "sentiment_cnn.pth")
        print("Validation loss improved, model saved.")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

print("Model saved successfully!")
