import torch
import torch.nn.functional as F
from torch import nn
from transformers.training.chunking import chunkText


def inference(decoder):
    tinyShakespeare = (
        open("../data/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
    )
    _, _, vocabularySize, char2idx, idx2char = chunkText(tinyShakespeare)
    decoder.eval()
    totalString = "ALEXANDRE:\nI do not agree"
    print(f"input:\n{totalString}")
    T = 1

    for _ in range(500):
        input = [char2idx[c] for c in totalString[-16:]]
        one_hot_initString = F.one_hot(
            torch.tensor(input).long(), vocabularySize
        ).float()

        next_char = torch.distributions.Categorical(
            probs=nn.Softmax()(decoder(one_hot_initString.squeeze(0))[0][-1]) / T
        ).sample()
        totalString = totalString + idx2char[next_char]

    print("Prediction:")
    print(totalString)
    print("---------------------------------------------")
