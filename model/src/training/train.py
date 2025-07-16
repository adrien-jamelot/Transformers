# """Training."""
import numpy as np
import torch
from torch import nn
from modules.decoder import Decoder
from training.utils.run_n_epochs import run_n_epochs
from training.utils.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F


def train():
    tinyShakespeare = (
        open("../data/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
    )
    inputs, targets, vocab_size, char2idx, idx2char = chunkText(tinyShakespeare)
    one_hot_examples = F.one_hot(torch.stack(inputs[:-1]).long(), vocab_size).float()
    loss_fn = nn.CrossEntropyLoss()
    model_parameters = {
        "nb_layers": 1,
        "vocabulary_size": vocab_size,
        "dim_model": 512,
        "dim_key": 64,
        "dim_value": 64,
        "nb_heads": 8,  # must be so that nb_heads * dim_value = dim_model else the concat won't make sense
    }

    decoder = Decoder(model_parameters)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3, betas=(0.9, 0.98))
    dataset = CustomDataset(one_hot_examples, targets)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    run_n_epochs(5, train_dataloader, decoder, loss_fn, optimizer, idx2char)


def chunkText(text):
    # unique characters in the file @ will be the initial character and # the final character.
    # They are not in the text.
    vocab = sorted(set(text + "@" + "#"))
    vocab_size = len(vocab)

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    int_text_tensor = torch.tensor(text_as_int)

    seq_length = 16

    examples_per_epoch = len(text) // (seq_length)
    chunks = torch.chunk(int_text_tensor, examples_per_epoch, 0)

    inputs = [chunk[:-1] for chunk in chunks]
    targets = [chunk[1:] for chunk in chunks]
    return inputs, targets, vocab_size, char2idx, idx2char
