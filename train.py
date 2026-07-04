import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from utils import build_vocab
from dataset import IMDBDataset, collate_fn
from model import SentimentRNN

df = pd.read_csv("data/imdb.csv")

# sentiment column should contain 0/1
vocab = build_vocab(df["review"])

with open("saved_model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

dataset = IMDBDataset(df, vocab)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=True, collate_fn=collate_fn)

model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for reviews, labels in loader:
        reviews = reviews.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(reviews)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "saved_model/model.pth")
print("Training Complete")
