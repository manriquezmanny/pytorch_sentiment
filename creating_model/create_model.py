import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, HfFolder

# Define the CNN model
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

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    predicted = (outputs >= 0.5).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy.item()

print(torch.cuda.is_available())

# Load preprocessed data and vocab
data = pd.read_pickle("encoded_data.pkl")
vocab = torch.load("vocab.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data for training
tweets = torch.stack(data["text"].tolist()).to(device)
labels = torch.tensor(data["target"].tolist()).to(device)
X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=12)

# Set hyperparameters
vocab_size = len(vocab)
embed_dim = 128
num_classes = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 32
accumulation_steps = 4
patience = 2
min_delta = 0.001

# Initialize model, loss function, and optimizer
model = SentimentCNN(vocab_size, embed_dim, num_classes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Initialize Hugging Face API
api = HfApi()
token = HfFolder.get_token()
repo_id = "manriquezmanny/pytorch-sentiment"

# Training loop with early stopping
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = len(X_train) // batch_size
    optimizer.zero_grad()

    for i in range(0, len(X_train), batch_size):
        batch_tweets = X_train[i:i+batch_size]
        batch_labels = y_train[i:i+batch_size].float()

        outputs = model(batch_tweets).squeeze(1)
        loss = criterion(outputs, batch_labels)
        loss.backward()

        if (i // batch_size + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        batch_accuracy = calculate_accuracy(outputs, batch_labels)
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{i//batch_size + 1}/{total_batches}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}", end="")

    epoch_loss /= total_batches
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze(1)
        val_loss = criterion(val_outputs, y_val.float())
        val_accuracy = calculate_accuracy(val_outputs, y_val)

    print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "sentiment_cnn.pth")

        # Upload model to Hugging Face
        api.upload_file(
            path_or_fileobj="sentiment_cnn.pth",
            path_in_repo="sentiment_cnn.pth",
            repo_id=repo_id,
            token=token
        )
        print("Validation loss improved, model saved and uploaded.")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

print("Model training completed.")