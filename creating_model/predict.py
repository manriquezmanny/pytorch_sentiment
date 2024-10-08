import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the SentimentCNN class
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.global_pool(x).squeeze(2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Load the vocabulary using torch.load with weights_only=True
vocab = torch.load("vocab.pth", weights_only=True)  # Load the vocabulary

# Initialize the model and load state dict
model = SentimentCNN(vocab_size=len(vocab), embedding_dim=128, num_classes=1).to(device)
state_dict = torch.load("sentiment_cnn.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Preprocess tweets
def preprocess_tweets(tweet):
    lowercase = tweet.lower()
    no_urls = re.sub(r'https?://\S+|www\.\S+', '', lowercase)
    no_html = re.sub(r'<.*?>', '', no_urls)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', no_html)
    return cleaned_text

# Encode tweets
def text_pipeline(vocab, input_text):
    preprocessed_text = preprocess_tweets(input_text)  # Helper function
    tokens = word_tokenize(preprocessed_text)  # Tokenization with nltk
    vocab_indices = [vocab.get(token, vocab['<unk>']) for token in tokens]  # Encoding text
    vocab_indices = vocab_indices[:100]  # Ensuring list doesn't exceed 100 elements
    tensor = torch.tensor(vocab_indices, dtype=torch.long).to(device)  # Turning indices into a tensor of type long
    tensor = F.pad(tensor, pad=(0, 100 - tensor.size(0)), value=vocab['<pad>'])  # Padding Tensor to ensure consistent element size
    return tensor.unsqueeze(0)

# Define the classify function
def classify_new_data(text):
    tensor = text_pipeline(vocab, text)
    with torch.no_grad():
        output = model(tensor)
    return 1 if output.item() >= 0.5 else 0

# Example usage
if __name__ == '__main__':
    sample_text = "I just bought a cookie!."
    prediction = classify_new_data(sample_text)
    print(f'Prediction: {prediction}')
