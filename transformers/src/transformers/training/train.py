# """Training."""
import numpy as np
import torch
from torch import nn
from transformers.blocks.decoder import Decoder
from transformers.training.run_n_epochs import run_n_epochs
from transformers.training.CustomDataset import CustomDataset
from transformers.training.chunking import chunkText
from transformers.training.utils import convertDictValuesToString
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchinfo import summary

import mlflow


def train():
    tinyShakespeare = (
        open("../data/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
    )
    inputs, targets, vocab_size, char2idx, idx2char = chunkText(tinyShakespeare)

    one_hot_examples = F.one_hot(
        torch.stack(inputs[:-1]).long(), vocab_size
    ).float()  # exclude last one which may be incomplete
    loss_fn = nn.CrossEntropyLoss()
    modelParameters = {
        "nb_layers": 1,
        "vocabulary_size": vocab_size,
        "dim_model": 512,
        "dim_key": 64,
        "dim_value": 64,
        "nb_heads": 8,  # must be so that nb_heads * dim_value = dim_model else the concat won't make sense
    }
    trainingParameters = {
        "epochs": 5,
        "learning_rate": 5e-3,
        "betas": (0.9, 0.98),
        "optimizer": "Adam",
        "batch_size": 64,
        "shuffle": True,
        "loss": loss_fn.__class__.__name__,
    }

    decoder = Decoder(modelParameters)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3, betas=(0.9, 0.98))
    dataset = CustomDataset(one_hot_examples, targets)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    with mlflow.start_run() as run:
        mlflow.log_params(convertDictValuesToString(trainingParameters))
        # mlflow.log_model_params(convertDictValuesToString(modelParameters))

        # Log model summary.
        with open("decoder_summary.txt", "w") as f:
            f.write(str(summary(decoder)))
            mlflow.log_artifact("decoder_summary.txt")

        run_n_epochs(
            trainingParameters["epochs"],
            train_dataloader,
            decoder,
            loss_fn,
            optimizer,
            idx2char,
        )
