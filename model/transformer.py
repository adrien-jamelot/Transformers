from torch import nn
import torch
import json
import simplejson

with open("""c:\\users\\adrie\\GraphDataScience\\Transformers
\\model\\transformer_config.json""") as config:
    model_parameters_default = simplejson.load(config)
    
# For positional encoding we use the torch implementation to avoid
# inefficient loops.

class Transformer(nn.Module, model_parameters=model_parameters_default):
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
        self.config = model_parameters
        self.encoder = Encoder(model_parameters["encoder"])
        self.decoder = Decoder(model_parameters["decoder"])
        self.first_decoder_output = ??

    def forward(self, x):
        """Apply a step forward."""
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class Encoder(nn.Module):
    """Encoder."""

    def __init__(self, encoderConfig):
        """Initialize."""
        super().__init__()
        self.nb_layers = encoderConfig["nb_layers"]
        self.dim_model = encoderConfig["dim_model"]
        self.norm = nn.LayerNorm(normalized_shape=self.dim_model,
                                 elementwise_affine=True, bias=True)
        self.multiheads = [MultiHeadAttention(encoderConfig["multihead"])
                           for i in range(self.nb_layers)]
        self.feedforwards = [nn.Linear(self.dim_model, self.dim_model)
                             for i in range(self.nb_layers)]

    def forward(self, x):
        """Forward."""
        for i in range(self.nb_layers):
            h1 = addAndNorm(x, self.multiheads[i](x, x, x), self.norm)
            x = addAndNorm(h1, self.feedforwards[i](h1), self.norm)
        return x


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self, decoderConfig):
        """Initialize."""
        super().__init__()
        self.dim_model = decoderConfig["dim_model"]
        self.layer = []
        self.multiheads1 = [MultiHeadAttention(decoderConfig["multihead"])
                            for i in range(self.nb_layers)]
        self.multiheads2 = [MultiHeadAttention(decoderConfig["multihead"])
                            for i in range(self.nb_layers)]
        self.feedforwards = [nn.Linear(self.dim_model, self.dim_model)
                             for i in range(self.nb_layers)]
        self.toProba = nn.Sequential(
            nn.Linear(self.nb_channels, self.dim_model),
            nn.Softmax()
        )

    def forward(self, input, lastOutput, encoderOutput):
        """Forward."""
        for i in range(self.nb_layers):
            h1 = self.AddAndNorm(lastOutput,
                                 self.multiheads1[i](lastOutput,
                                                     lastOutput,
                                                     lastOutput))
            h2 = self.AddAndNorm(h1,
                                 self.multiheads2[i](h1,
                                                     encoderOutput))
            lastOutput = addAndNorm(h2, self.feedforwards[i](h2))
        return self.toProba(lastOutput)


def ScaledDotProductAttention(Q, K, V, attention_parameters):
    """Scaled Dot-Product Attention.

    Inputs:
    - Q: query
    - K: key
    - V: value
    """
    dim_model = attention_parameters["dim_model"]
    return torch.matmul(torch.divide(torch.matmul(Q, K),
                                     torch.sqrt(dim_model)),
                        V)


class MultiHeadAttention(nn.Module, multi_head_config):
    """Multi-Head Attention.

    Inputs:
    - multi_head_config: dictionary
    """

    def __init__(self, multi_head_config):
        """Initialize multi-head."""
        super().__init__()
        self.dim_key = multi_head_config["layer"]["dim_key"]
        self.dim_model = multi_head_config["layer"]["dim_model"]
        self.nb_heads = multi_head_config["nb_heads"]

        self.WQs = [nn.Linear(self.dim_model, self.dim_key)
                    for i in range(self.nb_head)]
        self.WKs = [nn.Linear(self.dim_model, self.dim_key)
                    for i in range(self.nb_head)]
        self.WVs = [nn.Linear(self.dim_model, self.dim_value)
                    for i in range(self.nb_head)]

    def forward(self, Q, K, V):
        """One step of the multi-head block."""
        heads = [ScaledDotProductAttention(self.WQs[i](Q),
                                           self.WKs[i](K),
                                           self.WVs[i](V),
                                           self.dim_model)
                 for i in range(self.nb_heads)]
        return torch.cat([head for head in heads], 0)
