from transformers.training.train_loop import train_loop
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.modules.loss import _Loss
from torch import Tensor


def run_n_epochs(
    epochs: int,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: torch.optim.Optimizer,
    idx2char: list[str],
):
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer, t, idx2char)
