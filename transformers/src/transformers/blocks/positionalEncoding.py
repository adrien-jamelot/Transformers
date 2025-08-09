import math
import torch


def positionalEncoding(x):
    """Add positional encoding to a tensor.

    Only the default Attention is all you need version is supported:    
    .. math::
    PE_{(pos,2i)} &= \mathrm{sin}(\frac{\mathrm{pos}}{1000^{\frac{2i}{\mathrm{dim}_{model}}}}) \\
    PE_{(pos,2i+1)} &= \mathrm{cos}(\frac{\mathrm{pos}}{1000^{\frac{2i}{\mathrm{dim}_{model}}}})

    :param x: Input tensor, whose last two dimensions are interpreted respectively as
    (n-1)-th axis:  position
    n-th axis: embedding
    :returns: Input tensor with embedding values shifted according to the formula.
    """

    def shiftArgumentForSineOrCosine(embeddingIndex):
        """If the embeddingIndex is even keep as a sine, if it is odd shift by pi/2 so that it becomes a cosine (sin(alpha+pi/2) = cos(alpha).

        :param embeddingIndex: position of the embedding vector being read.

        """
        return math.pi / 2 * (embeddingIndex % 2 == 1)

    dim_model = x.shape[-1]
    values = [
        [
            shiftArgumentForSineOrCosine(embeddingIndex)
            + pos / math.pow(10000, 2 * (embeddingIndex // 2) / dim_model)
            for embeddingIndex in range(dim_model)
        ]
        for pos in range(x.shape[-2])
    ]
    return x + torch.sin(torch.tensor(values)).unsqueeze(0)
