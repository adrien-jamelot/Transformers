import torch
from torch import Tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from transformers.training.CustomDataset import customDatasetType
import mlflow


def runOneEpoch(
    trainingDataloader: DataLoader[customDatasetType],
    validationDataloader: DataLoader[customDatasetType],
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: Optimizer,
    epoch: int,
    indexToCharacterMap: list[str],
    modelName: str,
):
    print(f"Running epoch {epoch}")
    datasetSize = len(trainingDataloader.dataset)
    batchSize = trainingDataloader.batch_size
    nbBatches = len(trainingDataloader)
    model.train()
    logFrequency = 100
    for batchIndex, (X, y) in enumerate(trainingDataloader):
        pred, loss = trainingStep(X, y, model, loss_fn, optimizer)
        if batchIndex % logFrequency == 0:
            lossValue = loss.item()
            printTrainingStatus(
                X[0],
                y[0],
                pred[0],
                indexToCharacterMap,
                lossValue,
                batchIndex,
                datasetSize,
                batchSize,
            )
            mlflow.log_metric(
                key="Training Loss",
                value=lossValue,
                step=(
                    (nbBatches // logFrequency + 1) * logFrequency * epoch + batchIndex
                )
                // logFrequency,
            )

    validationLoss = computeLoss(
        dataloader=validationDataloader, model=model, loss_fn=loss_fn
    )
    mlflow.log_metric(key="Validation Loss", value=validationLoss, step=epoch)


def trainingStep(
    X: Tensor, y: Tensor, model: nn.Module, loss_fn: _Loss, optimizer: Optimizer
) -> tuple[Tensor, Tensor]:
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred.transpose(-2, -1), y)
    # Backpropagation
    loss.backward()
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    return pred, loss


@torch.no_grad()
def computeLoss(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: _Loss,
) -> float:
    model.eval()
    validationLoss = 0
    for X, y in dataloader:
        validationLoss += loss_fn(model(X).transpose(-2, -1), y)
    return validationLoss / len(dataloader)


def printTrainingStatus(
    X: Tensor,
    y: Tensor,
    prediction: Tensor,
    indexToCharacterMap: list[str],
    lossValue: Number,
    batch: int,
    datasetSize: int,
    batchSize: int,
):
    currentExample = batch * batchSize + 1
    inputString = torch.argmax(X, dim=1).tolist()
    predictionString = torch.argmax(prediction, dim=1).tolist()

    print(f"loss: {lossValue:>7f}  [{currentExample:>5d}/{datasetSize:>5d}]")
    print("Input      :" + "".join([indexToCharacterMap[i] for i in inputString]))
    print("Target     :" + "".join([indexToCharacterMap[i] for i in y]))
    print("Prediction :" + "".join([indexToCharacterMap[i] for i in predictionString]))
