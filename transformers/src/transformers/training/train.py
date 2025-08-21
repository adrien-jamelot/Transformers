# """Training."""
from mlflow import artifacts
import torch
from torch import nn
from transformers.blocks.decoder import Decoder
from transformers.training.runNEpochs import runNEpochs
from transformers.training.CustomDataset import CustomDataset, customDatasetType
from transformers.training.chunking import chunkText, saveTokenisationInfo, encodeChunks
from transformers.training.utils import convertDictValuesToString
from torch.utils.data import random_split
from transformers.training.TrainingModels import (
    SplittingRatios,
    GlobalParameters,
    ModelParameters,
    TrainingParameters,
    TrainingDatasets,
)
from torchinfo import summary
import mlflow.pytorch as mlpt
import mlflow
from transformers.training.runOneEpoch import computeLoss
from torch.utils.data import DataLoader


def train():
    tinyShakespeare = (
        open("../data/raw/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
    )
    chunks = chunkText(tinyShakespeare, nbCharactersPerChunk=16)
    saveTokenisationInfo(chunks, "../data/chunks/chunks.pkl")
    X = encodeChunks(chunks)
    loss_fn = nn.CrossEntropyLoss()

    globalParameters = GlobalParameters(randomSeed=42)

    modelParameters = ModelParameters(
        nbLayers=1,
        dimModel=512,
        dimKey=64,
        dimValue=64,
        nbHeads=8,
        vocabularySize=chunks.vocabSize,
    )

    splittingRatios = SplittingRatios(
        trainingRatio=0.8, validationRatio=0.15, testingRatio=0.05
    )

    dataset = CustomDataset(X, chunks.targets)
    trainingParameters = TrainingParameters(
        epochs=15,
        learningRate=5e-3,
        betas=(0.9, 0.8),
        optimizer="Adam",
        batchSize=64,
        shuffle=True,
        loss=loss_fn.__class__.__name__,
        splittingRatios=splittingRatios,
        trainingSize=int(splittingRatios.trainingRatio * len(dataset)),
        validationSize=int(splittingRatios.validationRatio * len(dataset)),
        testingSize=int(splittingRatios.testingRatio * len(dataset)),
    )

    trainingDataset, validationDataset, testingDataset = random_split(
        dataset,
        [
            splittingRatios.trainingRatio,
            splittingRatios.validationRatio,
            splittingRatios.testingRatio,
        ],
        torch.Generator().manual_seed(globalParameters.randomSeed),
    )

    trainingDatasets = TrainingDatasets(
        trainingDataset=trainingDataset.dataset,
        validationDataset=validationDataset.dataset,
        testingDataset=testingDataset.dataset,
    )
    decoder = Decoder(modelParameters)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=trainingParameters.learningRate,
        betas=trainingParameters.betas,
    )
    with mlflow.start_run():
        mlflow.log_params(convertDictValuesToString(trainingParameters.model_dump()))
        # Log model summary.
        with open("decoder_summary.txt", "w") as f:
            f.write(str(summary(decoder)))
            mlflow.log_artifact("decoder_summary.txt")

        runNEpochs(
            trainingDatasets=trainingDatasets,
            trainingParameters=trainingParameters,
            model=decoder,
            loss_fn=loss_fn,
            optimizer=optimizer,
            idx2char=chunks.idx2char,
        )
        testingDataloader = DataLoader(
            trainingDatasets.testingDataset, batch_size=64, shuffle=True
        )
        testingLoss = computeLoss(
            dataloader=testingDataloader, model=decoder, loss_fn=loss_fn
        )
        mlflow.log_metric(key="Testing Loss", value=testingLoss)
        mlflow.log_artifact(local_path="../data/chunks/chunks.pkl")
        mlpt.log_model(
            decoder,
            name="decoder",
            input_example=X[0].numpy(),
        )
