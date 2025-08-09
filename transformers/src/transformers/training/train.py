# """Training."""
import numpy as np
import torch
from torch import nn
from transformers.blocks.decoder import Decoder
from transformers.training.run_n_epochs import run_n_epochs
from transformers.training.CustomDataset import CustomDataset
from transformers.training.chunking import chunkText
from torch.utils.data import DataLoader
from torch.nn import functional as F


def train():
    tinyShakespeare = (
        open("../data/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
    )
    inputs, targets, vocab_size, _, idx2char = chunkText(tinyShakespeare)
    one_hot_examples = F.one_hot(
        torch.stack(inputs[:-1]).long(), vocab_size
    ).float()  # exclude last one which may be incomplete
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
