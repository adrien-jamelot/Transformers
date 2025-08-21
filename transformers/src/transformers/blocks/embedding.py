from torch import nn
from transformers.training.TrainingModels import ModelParameters


class Embedding(nn.Module):
    """Embedding."""

    def __init__(self, modelParameters: ModelParameters):
        """Initialize embedding."""
        super().__init__()
        self.embedding = nn.Linear(
            modelParameters.vocabularySize, modelParameters.dimModel
        )

    def forward(self, x):
        """Forward step."""
        return self.embedding(x)
