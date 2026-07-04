import torch
from torch.utils.data import Dataset
from utils import text_to_indices

class IMDBDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = text_to_indices(self.df.iloc[idx]["review"], self.vocab)
        label = int(self.df.iloc[idx]["sentiment"])
        return torch.tensor(review), torch.tensor(label)

def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews = torch.nn.utils.rnn.pad_sequence(
        reviews, batch_first=True, padding_value=0
    )
    labels = torch.stack(labels)
    return reviews, labels
