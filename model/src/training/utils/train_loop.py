import torch


def train_loop(dataloader, model, loss_fn, optimizer, epoch, indexToCharacterMap):
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
            logTrainingStatus(indexToCharacterMap, loss, batch, pred, datasetSize)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"model_epoch_{epoch}.pt",
    )


def probabilitiesMatrixToString(X, characterForIndex):
    mostLikelyCharacterIndexSequence = torch.max(X[0], dim=1)[1].toList()
    return "".join([characterForIndex[i] for i in mostLikelyCharacterIndexSequence])


def logTrainingStatus(indexToCharacterMap, loss, batch, prediction, datasetSize):
    loss, current = loss.item(), batch * 64 + len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{datasetSize:>5d}]")
    print(
        "Input      :"
        + "".join([indexToCharacterMap[i] for i in torch.max(X[0], dim=1)[1].tolist()])
    )
    print("Target     :" + "".join([indexToCharacterMap[i] for i in y[0]]))
    print(
        "Prediction :"
        + "".join(
            [indexToCharacterMap[i] for i in torch.max(pred[0], dim=1)[1].tolist()]
        )
    )
