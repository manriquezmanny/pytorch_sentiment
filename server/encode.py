import torch
import pandas as pd
import re
import torch.nn.functional as F
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator


# Helper function to preprocess tweets. Lower method and Regex to remove unwanted characters.
def preprocess_tweets(tweet):
    lowercase = tweet.lower()
    no_urls = re.sub(r'https?://\S+|www\.\S+', '', lowercase)
    no_html = re.sub(r'<.*?>', '', no_urls)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', no_html)
    return cleaned_text


# Function to encode tweets and turn them into pytorch tensors.
def text_pipeline(vocab, input_text):
    preprocessed_text = preprocess_tweets(input_text) # Helper function
    tokens = preprocessed_text.split() # Splitting text by space.
    vocab_indicies = [vocab[token] for token in tokens] # Encoding text
    vocab_indicies = vocab_indicies[:100] # Ensuring list doesn't exceed 100 elements.
    tensor = torch.tensor(vocab_indicies, dtype=torch.long) # Turning indicies into a tensor of type long.
    tensor = F.pad(tensor, pad=(0, 100 - tensor.size(0)), value=vocab['<pad>']) # Padding Tensor to ensure consistent element size.
    return tensor


# Loading ONLY the tweets and labels from sentiment140 csv file.
data = pd.read_csv("./sentiment140.csv", header=None, encoding="ISO-8859-1")
data = data[[0, 5]]
data.columns = ["target", "text"]
tweets = data["text"]
labels = data["target"]
data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)
print("Loaded Data")

# Instantiating a Counter object to count word frequency using python's standard collections library.
counter = Counter()
# Updating counter with each tweet.
for text in data["text"]:
    text = preprocess_tweets(text)
    counter.update(text)
most_common_words = counter.most_common(50000)
counter = Counter(dict(most_common_words))
print("Created Counter")


# Instantiating a Vocab object to filter out words. Keeping only 50,000 most used words.
def yield_tokens(data_iter):
    for text in data_iter:
        yield preprocess_tweets(text).split()

vocab = build_vocab_from_iterator(yield_tokens(data["text"]), specials=['<pad>'])
vocab.set_default_index(vocab['<pad>'])


# Encoding and converting text data to tensors with text_pipeline function.
for i in range(len(data["text"])):
    text = data.loc[i, 'text']
    tensor = text_pipeline(vocab, text)
    data.at[i, 'text'] = tensor
    print(f"\r{i+1} out of {len(data)} tweets encoded", end='')
print("Created Vocab Map")


# Saving encoded data and vocab mapping.
data.to_pickle("encoded_data.pkl")
torch.save(vocab, "vocab.pth")
print("Saved Files")
print(f"\n {data["text"]}")