import torch
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from huggingface_hub import HfApi, HfFolder

# Download nltk data
nltk.download('punkt')

# Helper function to preprocess tweets
def preprocess_tweets(tweet):
    lowercase = tweet.lower()
    no_urls = re.sub(r'https?://\S+|www\.\S+', '', lowercase)
    no_html = re.sub(r'<.*?>', '', no_urls)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', no_html)
    return cleaned_text

# Load data
data = pd.read_csv("./sentiment140.csv", header=None, encoding="ISO-8859-1")
data = data[[0, 5]]
data.columns = ["target", "text"]
data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)
tweets = data["text"]

# Build the vocabulary
all_tokens = []
for text in tweets:
    text = preprocess_tweets(text)
    all_tokens.extend(word_tokenize(text))

counter = Counter(all_tokens)
most_common_words = counter.most_common(50000)
vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}
vocab['<unk>'] = len(vocab)
vocab['<pad>'] = len(vocab)

# Save vocab locally
torch.save(vocab, "vocab.pth")

# Upload vocab to Hugging Face
api = HfApi()
token = HfFolder.get_token()
api.upload_file(
    path_or_fileobj="vocab.pth",
    path_in_repo="vocab.pth",
    repo_id="manriquezmanny/pytorch-sentiment",
    token=token
)

print("Vocabulary saved and uploaded to Hugging Face.")
