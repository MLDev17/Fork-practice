import re
from collections import Counter
from nltk.tokenize import word_tokenize

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return word_tokenize(text)

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    vocab = {"<PAD>":0, "<UNK>":1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def text_to_indices(text, vocab):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
