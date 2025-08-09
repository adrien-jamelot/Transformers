import torch
import torch.nn.functional as F
from torch import nn


def inference(decoder, model_parameters, char2idx, idx2char):
    decoder.eval()
    totalString = "ALEXANDRE:\nI do not agree"
    print(f"input:\n{totalString}")
    T = 1
    for _ in range(500):
        input = [char2idx[c] for c in totalString[-16:]]
        one_hot_initString = F.one_hot(
            torch.tensor(input).long(), model_parameters["vocabulary_size"]
        ).float()
        # print(decoder(one_hot_initString.squeeze(0))[0][-1].shape)
        next_char = torch.distributions.Categorical(
            probs=nn.Softmax()(decoder(one_hot_initString.squeeze(0))[0][-1]) / T
        ).sample()
        totalString = totalString + idx2char[next_char]

    # print(one_hot_initString.shape)
    print("Prediction:")
    print(
        totalString
    )  #''.join([idx2char[i] for i in torch.max(decoder(one_hot_initString.squeeze(0))[0],1)[1].tolist()]))
    print("---------------------------------------------")
