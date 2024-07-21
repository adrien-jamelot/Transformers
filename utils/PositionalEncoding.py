"""Positional Encoding."""
import math
import torch


def positionalEncoding(x, dim_model):
    """Positional Encoding."""
    def sineOrCosine(i):
        """sin(alpha+pi/2) = cos(alpha)."""
        return math.pi/2*(i % 2 == 1)
    values = [
        [sineOrCosine(i) + pos/math.pow(10000, 2*(i//2)/dim_model) for
         i in range(dim_model)]
        for pos in range(x.shape[0])
    ]
    return torch.sin(torch.tensor(values))
