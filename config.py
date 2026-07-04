import torch

BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_EPOCHS = 10
LR = 1e-3
MAX_LEN = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
