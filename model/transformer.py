from torch import nn
import torch
import simplejson
import sys
sys.path.append('..\\..')
from Transformers.utils.PositionalEncoding import positionalEncoding
from Transformers.utils.addAndNorm import addAndNorm

with open("""..\\model\\transformer_config.json""") as config:
    model_parameters_default = simplejson.load(config)


class Transformer(nn.Module):
    """Main transformer block.

    Inputs: - model_parameters: ordered dictionary of key-values describing the
    layer parameters of the model:
      - dim_model: dimension of the model.
      - layers: dictionary of key-values describing specific layers
        - <layer_name>: dictionary of parameters for the specific multi-head
          layer
          - attention: dictionary of parameters for the specific attention
            function
            - dim_key: dimension of the key and query.
            - dim_value: dimension of the value.
          - nb_head: number of heads.

    """

    def __init__(self, model_parameters):
        """Initialize parameters."""
        super().__init__()
        self.dim_model = model_parameters["dim_model"]
        self.encoder = Encoder(model_parameters["encoder"])
        self.decoder = Decoder(model_parameters["decoder"])
        self.embedding = Embedding(model_parameters)
        self.toProba = nn.Sequential(
            nn.Linear(self.dim_model,
                      model_parameters["vocabulary_size"]),
            nn.Softmax()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, lastOutput):
        """Apply a step forward."""
        encoderInput = self.embedding(x) + positionalEncoding(
            x, self.dim_model)
        decoderInput = self.embedding(lastOutput) + positionalEncoding(
            lastOutput, self.dim_model)
        encoderInput = self.dropout(encoderInput)
        decoderInput = self.dropout(decoderInput)
        encoderOutput = self.encoder(encoderInput)
        decoderOutput = self.decoder(decoderInput, encoderOutput)
        lastOutput = self.toProba(decoderOutput)
        return lastOutput


class Encoder(nn.Module):
    """Encoder."""

    def __init__(self, encoderConfig):
        """Initialize."""
        super().__init__()
        self.nb_layers = encoderConfig["nb_layers"]
        self.dim_model = encoderConfig["dim_model"]
        self.dim_feedforward = encoderConfig["feedforward"]["dim_feedforward"]
        self.norm = nn.LayerNorm(normalized_shape=self.dim_model,
                                 elementwise_affine=True, bias=True)
        self.multiheads = [MultiHeadAttention(encoderConfig["multihead"])
                           for i in range(self.nb_layers)]
        self.feedforwards = [nn.Sequential(nn.Linear(self.dim_model,
                                                     self.dim_feedforward),
                                           nn.ReLU(),
                                           nn.Linear(self.dim_feedforward,
                                                     self.dim_model))
                             for i in range(self.nb_layers)]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward."""
        for i in range(self.nb_layers):
            h1 = addAndNorm(x, self.dropout(self.multiheads[i](x, x,
                                                               x)),
                            self.norm)
            x = addAndNorm(h1, self.dropout(self.feedforwards[i](h1)),
                           self.norm)
        return x


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self, decoderConfig):
        """Initialize."""
        super().__init__()
        self.dim_model = decoderConfig["dim_model"]
        self.norm = nn.LayerNorm(normalized_shape=self.dim_model)
        self.dim_feedforward = decoderConfig["feedforward"]["dim_feedforward"]
        self.nb_layers = decoderConfig["nb_layers"]
        self.layer = []
        self.multiheads1 = [MultiHeadAttention(decoderConfig["multihead"])
                            for i in range(self.nb_layers)]
        self.multiheads2 = [MultiHeadAttention(decoderConfig["multihead"])
                            for i in range(self.nb_layers)]
        self.feedforwards = [nn.Sequential(nn.Linear(self.dim_model,
                                                     self.dim_feedforward),
                                           nn.ReLU(),
                                           nn.Linear(self.dim_feedforward,
                                                     self.dim_model))
                             for i in range(self.nb_layers)]

    def forward(self, decoderInput, encoderOutput):
        """Forward."""
        for i in range(self.nb_layers):
            h1 = addAndNorm(decoderInput,
                            self.multiheads1[i](decoderInput,
                                                decoderInput,
                                                decoderInput),
                            self.norm)
            h2 = addAndNorm(h1,
                            self.multiheads2[i](h1,
                                                encoderOutput,
                                                encoderOutput),
                            self.norm)
            lastOutput = addAndNorm(h2,
                                    self.feedforwards[i](h2),
                                    self.norm)
        return lastOutput


def ScaledDotProductAttention(Q, K, V, dim_model):
    """Scaled Dot-Product Attention.

    Inputs:
    - Q: query
    - K: key
    - V: value
    - dim_model: model dimension
    """
    return torch.matmul(torch.divide(torch.matmul(Q, K.transpose(0, 1)),
                                     torch.sqrt(torch.Tensor([dim_model]))),
                        V)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Inputs:
    - multi_head_config: dictionary
    """

    def __init__(self, multi_head_config):
        """Initialize multi-head."""
        super().__init__()
        self.dim_key = multi_head_config["attention"]["dim_key"]
        self.dim_value = multi_head_config["attention"]["dim_value"]
        self.nb_heads = multi_head_config["nb_heads"]
        self.dim_model = self.dim_key * self.nb_heads

        self.WQs = [nn.Linear(self.dim_model, self.dim_key)
                    for i in range(self.nb_heads)]
        self.WKs = [nn.Linear(self.dim_model, self.dim_key)
                    for i in range(self.nb_heads)]
        self.WVs = [nn.Linear(self.dim_model, self.dim_value)
                    for i in range(self.nb_heads)]

    def forward(self, Q, K, V):
        """One step of the multi-head block."""
        heads = [ScaledDotProductAttention(self.WQs[i](Q),
                                           self.WKs[i](K),
                                           self.WVs[i](V),
                                           self.dim_model)
                 for i in range(self.nb_heads)]
        return torch.cat([head for head in heads], 1)


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
