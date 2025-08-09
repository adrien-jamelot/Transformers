import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from transformers.training.CustomDataset import CustomDataset


def train_loop(
    dataloader: DataLoader[CustomDataset],
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: Optimizer,
    epoch: int,
    indexToCharacterMap: list[str],
):
    """Train loop. Taken from pytorch tutorial."""

    datasetSize = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.transpose(1, 2), y)
        # Backpropagation
        loss.backward()
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            logTrainingStatus(
                X[0], y[0], pred[0], indexToCharacterMap, loss, batch, datasetSize
            )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        },
        f"model_epoch_{epoch}.pt",
    )


def probabilitiesMatrixToString(X, characterForIndex):
    mostLikelyCharacterIndexSequence = torch.max(X[0], dim=1)[1].toList()
    return "".join([characterForIndex[i] for i in mostLikelyCharacterIndexSequence])


def logTrainingStatus(
    X: Tensor,
    y: Tensor,
    prediction: Tensor,
    indexToCharacterMap: list[str],
    loss: Tensor,
    batch: int,
    datasetSize: int,
):
    lossValue = loss.item()
    currentExample = batch * 64 + 1

    inputString = torch.argmax(X, dim=1).tolist()
    predictionString = torch.argmax(prediction, dim=1).tolist()

    print(f"loss: {lossValue:>7f}  [{currentExample:>5d}/{datasetSize:>5d}]")
    print("Input      :" + "".join([indexToCharacterMap[i] for i in inputString]))
    print("Target     :" + "".join([indexToCharacterMap[i] for i in y]))
    print("Prediction :" + "".join([indexToCharacterMap[i] for i in predictionString]))
