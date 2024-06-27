from torch import nn
import torch


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

    def forward(self, x):
        """Apply a step forward."""
        for layer in self.config["encoder"]:
            
            h = MultiHeadAttention(layer["nb_head"])
        return h



class AddAndNorm(nn.Module, dim_model, block):
    """Residual connection."""
    def __init__(self, dim_model, block):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim_model,
                                 elementwise_affine=True,
                                 bias=True)
        self.block = block
    def forward(self, x):
        return self.norm(x + self.block(x))


class Encoder(nn.Module):
    """Encoder.
    """

    def __init__(self, encoderConfig):
        super().__init__()
        self.dim_model = encoderConfig["dim_model"]
        self.layer = []
        for i in range(encoderConfig["nb_layers"]):
            self.layer.append(
                ('multihead' + str(i),
                AddAndNorm(self.dim_model,
                           MultiHeadAttention(encoderConfig["multihead"])
                           )
                 )
            )
            self.layer.append(
                ('feedforward' + str(i),
                AddAndNorm(self.dim_model,
                           nn.Linear(self.dim_model, self.dim_model)
                           )
                 )
            )
        self.update = nn.Sequential(
            *self.layer
        )

    def forward(self, x):
        return self.update(x)


class Decoder(nn.Module):
    """Decoder.
    """

    def __init__(self, encoderConfig):
        super().__init__()
        self.dim_model = encoderConfig["dim_model"]
        self.layer = []
        for i in range(encoderConfig["nb_layers"]):
            self.layer.append(
                ('multihead' + str(i),
                 AddAndNorm(self.dim_model,
                            MultiHeadAttention(encoderConfig["multihead"])
                            )
                 )
            )
            self.layer.append(
                ('feedforward' + str(i),
                 AddAndNorm(self.dim_model,
                            nn.Linear(self.dim_model, self.dim_model)
                            )
                 )
            )
        self.update = nn.Sequential(
            *self.layer
        )

    def forward(self, x):
        return self.update(x)

class ScaledDotProductAttention(nn.Module, dim_model, attention_parameters):
    """Scaled Dot-Product Attention.

    Inputs:
    - dim_model: number of input channels that the model treats at once.
    - attention_parameters: dictionary of key-values describing the attention
    dimensions as:
    - dim_key: dimension of the key and query
    - dim_value: dimension of the value.

    """

    def __init__(self, dim_model, attention_parameters):
        """Initialize parameters."""
        super().__init__()
        self.dim_key = attention_parameters["dim_key"]
        self.dim_value = attention_parameters["dim_value"]
        self.K = nn.Linear(self.dim_key, dim_model)  # TODO Check dimensions
        self.V = nn.Linear(self.dim_value, dim_model)  # TODO
        self.activation = nn.Softmax()
        self.sqrt_dim_model = torch.sqrt(dim_model)

        def forward(self, x):
            """Forward Pass."""
            sdpa = torch.divide(self.K(x), self.sqrt_dim_model)
            # sdpa = mask(sdpa)
            sdpa = self.activation(sdpa)
            return self.V(sdpa)

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
        self.nb_head = multi_head_config["nb_head"]
        self.heads = [
            nn.Sequential(
                nn.Linear(self.dim_key, self.dim_model),
                ScaledDotProductAttention(
                    dim_model=self.dim_model,
                    attention_parameters=multi_head_config["layer"]
                )
            )
            for i in range(self.nb_head)
        ]
        
    def forward(self, x):
        """One step of the multi-head block."""
        return torch.cat([head(x) for head in self.heads], 0)
