from torch import nn

class Embedding(nn.Module):
    """Embedding."""

    def __init__(self, model_parameters):
        """Initialize embedding."""
        super().__init__()
        self.embedding = nn.Linear(model_parameters["vocabulary_size"],
                                   model_parameters["dim_model"])

    def forward(self, x):
        """Forward step."""
        return self.embedding(x)
