import pickle
import torch

from config import *
from model import SentimentRNN
from utils import text_to_indices

with open("saved_model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("saved_model/model.pth",
                                 map_location=DEVICE))
model.eval()

text = input("Enter Review: ")

indices = text_to_indices(text, vocab)
tensor = torch.tensor(indices).unsqueeze(0)

with torch.no_grad():
    pred = model(tensor).argmax(dim=1).item()

print("Positive" if pred == 1 else "Negative")
