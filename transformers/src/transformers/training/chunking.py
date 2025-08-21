from typing import List, Any
import torch
from torch.nn import functional as F
import numpy as np
from torch.types import Tensor
import pickle
import os
from pydantic import BaseModel


class TokenisationInfo(BaseModel):
    vocabSize: int
    char2idx: dict[str, int]
    idx2char: list[str]


class Chunk(BaseModel):
    inputs: List[Tensor]
    targets: List[Tensor]
    vocabSize: int
    char2idx: dict[str, int]
    idx2char: list[str]

    class Config:
        arbitrary_types_allowed = True


def chunkText(text: str, nbCharactersPerChunk: int) -> Chunk:
    # @ will be the initial character and # the final character.
    vocab = sorted(set(text + "@" + "#"))
    vocabSize = len(vocab)

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = vocab

    textAsInt = np.array([char2idx[c] for c in text])
    intTextTensor = torch.tensor(textAsInt)

    nbChunks = len(text) // (nbCharactersPerChunk)
    chunks = torch.chunk(intTextTensor, chunks=nbChunks, dim=0)

    inputs = [chunk[:-1] for chunk in chunks]
    targets = [chunk[1:] for chunk in chunks]

    return Chunk(
        inputs=inputs,
        targets=targets,
        vocabSize=vocabSize,
        char2idx=char2idx,
        idx2char=idx2char,
    )


def saveObject(obj: Any, filepath: str) -> None:
    directoryName = os.path.dirname(filepath)
    if not os.path.isdir(directoryName):
        os.makedirs(os.path.dirname(directoryName), exist_ok=True)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def saveTokenisationInfo(chunks: Chunk, filepath: str):
    tokenisationInfo = TokenisationInfo(
        vocabSize=chunks.vocabSize, char2idx=chunks.char2idx, idx2char=chunks.idx2char
    )
    saveObject(tokenisationInfo, filepath)


def readChunks(filepath: str) -> TokenisationInfo:
    with open(filepath, "rb") as file:
        return pickle.load(file)


def encodeChunks(chunks: Chunk) -> Tensor:
    oneHotInputs = F.one_hot(
        torch.stack(chunks.inputs[:-1]).long(), chunks.vocabSize
    ).float()  # exclude last one which may be incomplete
    return oneHotInputs
