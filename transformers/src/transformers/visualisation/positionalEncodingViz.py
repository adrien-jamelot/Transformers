import matplotlib.pyplot as plt
import torch
from transformers.blocks.positionalEncoding import positionalEncoding


def positionalEncodingViz():
    """Checking that the encoded position looks like something."""
    seqLength = 20
    embeddingDimension = 1024
    x = torch.ones((1, seqLength, embeddingDimension))
    posEncoding = positionalEncoding(x)
    for i in range(seqLength):
        plt.plot(posEncoding[:, i])
    plt.show()
