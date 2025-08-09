from typing import List, Tuple
import torch
import numpy as np
from torch.types import Tensor


def chunkText(
    text: str,
) -> Tuple[List[Tensor], List[Tensor], int, dict[str, int], list[str]]:
    # @ will be the initial character and # the final character.
    vocab = sorted(set(text + "@" + "#"))
    vocabSize = len(vocab)

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = vocab

    textAsInt = np.array([char2idx[c] for c in text])
    intTextTensor = torch.tensor(textAsInt)

    seqLength = 16
    nbChunks = len(text) // (seqLength)
    chunks = torch.chunk(intTextTensor, chunks=nbChunks, dim=0)

    inputs = [chunk[:-1] for chunk in chunks]
    targets = [chunk[1:] for chunk in chunks]
    return inputs, targets, vocabSize, char2idx, idx2char
