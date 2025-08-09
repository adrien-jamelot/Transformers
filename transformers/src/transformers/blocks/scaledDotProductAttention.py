import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, dim_model, masked=False):
        """Initialize.

        Inputs:
        - dim_model: model dimension
        - masked: prevents tokens to attend to the following ones.
        """
        super().__init__()
        self.dim_model = dim_model
        self.masked = masked
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        """Forward.

        Inputs:
        - Q: query
        - K: key
        - V: value
        """
        matmul_0 = torch.matmul(Q, K.transpose(-2, -1))
        scaled = torch.divide(matmul_0, torch.Tensor([self.dim_model]))
        if self.masked:
            mask = torch.ones(scaled.shape)
            mask = mask - torch.tril(mask)*mask
            mask = torch.where(mask == 1, float('-inf'), 0)
            scaled = scaled + mask
        softmaxed = self.softmax(scaled)
        sdpa = torch.matmul(softmaxed, V)
        return sdpa
