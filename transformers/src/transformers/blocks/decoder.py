from torch import nn
from .addAndNorm import addAndNorm
from .positionalEncoding import positionalEncoding
from .multiHeadAttention import MultiHeadAttention
from .embedding import Embedding


class Decoder(nn.Module):
    """A simple decoder taken from Attention is all you need.
    It does not
    """

    def __init__(self, model_parameters):
        """Initialize."""
        super().__init__()
        self.dim_model = model_parameters["dim_model"]
        self.norm = nn.LayerNorm(normalized_shape=self.dim_model)
        self.dim_feedforward = model_parameters[
            "dim_model"
        ]  # must be same dimension because we add residuals
        self.nb_layers = model_parameters["nb_layers"]
        self.embedding = Embedding(model_parameters)
        self.multiheads1 = nn.ModuleList(
            [
                MultiHeadAttention(model_parameters, masked=True)
                for i in range(self.nb_layers)
            ]
        )
        # self.multiheads2 = nn.ModuleList([MultiHeadAttention(model_parameters["multihead"],
        #                                       masked=False)
        #                    for i in range(self.nb_layers)])
        self.feedforwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim_model, self.dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(self.dim_feedforward, self.dim_model),
                )
                for i in range(self.nb_layers)
            ]
        )
        self.toLogit = nn.Linear(self.dim_model, model_parameters["vocabulary_size"])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass of the decoder.

        :param x: Input tensor

        """
        x = positionalEncoding(self.embedding(x))
        x = self.dropout(x)
        for i in range(self.nb_layers):
            h1 = addAndNorm(x, self.dropout(self.multiheads1[i](x, x, x)), self.norm)
            # h2 = addAndNorm(h1,
            #                self.dropout(self.multiheads2[i](h1, h1, h1)),
            #                self.norm)
            feedforwardOutput = self.feedforwards[i](h1)
            layerOutput = addAndNorm(h1, self.dropout(feedforwardOutput), self.norm)
            x = layerOutput
        logits = self.toLogit(x)
        return logits
