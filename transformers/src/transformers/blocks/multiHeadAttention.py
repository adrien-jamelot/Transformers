from torch import nn
import torch
from transformers.training.TrainingModels import ModelParameters
from .scaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Inputs:
    - multi_head_config: dictionary
    """

    def __init__(self, multi_head_config: ModelParameters, masked=False):
        """Initialize multi-head."""
        super().__init__()
        self.dim_key = multi_head_config.dimKey
        self.dim_value = multi_head_config.dimValue
        self.nb_heads = multi_head_config.nbHeads
        self.dim_model = self.dim_key * self.nb_heads

        self.WQs = nn.ModuleList(
            [nn.Linear(self.dim_model, self.dim_key) for _ in range(self.nb_heads)]
        )
        self.WKs = nn.ModuleList(
            [nn.Linear(self.dim_model, self.dim_key) for _ in range(self.nb_heads)]
        )
        self.WVs = nn.ModuleList(
            [nn.Linear(self.dim_model, self.dim_value) for _ in range(self.nb_heads)]
        )
        self.spda = ScaledDotProductAttention(self.dim_model, masked)

    def forward(self, Q, K, V):
        """One step of the multi-head block."""
        heads = [
            self.spda(self.WQs[i](Q), self.WKs[i](K), self.WVs[i](V))
            for i in range(self.nb_heads)
        ]
        return torch.cat([head for head in heads], -1)
