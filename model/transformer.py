from torch import nn
import torch


model_parameters_default = {
    "batch_size": 512,
    "encoder": {
        "nb_layers": 6,
        "blocks": {
            "multihead": {
                    "attention": {
                        "dim_model": 512,
                        "dim_key": 64,
                        "dim_value": 64
                    },
                    "nb_head": 8
                },
            "feedforward": {
                    "dim_input": ?,
                    "dim_output": ?
                }
            }
    },
    "decoder": {
        "layers": {
            "1": {
                "attention": {
                    "masked": "true",
                    "dim_model": 512,
                    "dim_key": 64,
                    "dim_value": 64
                },
                "nb_head": 8
            },
            "2": {
                "attention": {
                    "masked": "false",
                    "dim_model": 512,
                    "dim_key": 64,
                    "dim_value": 64,
                },
                "nb_head": 8
            }
        }
    }
}


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


class Encoder(nn.Module, encoderConfig):
    """Encoder.
    """
    def __init__(self, encoderConfig):
        super().__init__()
        self.update = nn.Sequential(
            [AddAndNorm(MultiHeadAttention(encoderConfig["multihead"])),
             AddAndNorm(nn.Linear()),
             for i in range(encoderConfig["nbLayer"])]
        )
            
    def forward(self, x):
        return self.update(x)

        
class AddAndNorm(nn.Module, block):
    """Residual connection.
    """
    def __init__(self):
        super().__init__()
        self.norm = ?
    def forward(self, x):
        return self.norm(x + block(x))

    
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
        self.update = nn.Sequential(
            self.K,
            torch.div(sqrt(dim_model))  # TODO make precise
            self.activation,
        )

    def forward(self, x):
        """Forward Pass."""
        sdpa = torch.matmul(x, self.K)
        sdpa = mask(sdpa)
        sdpa = self.activation(sdpa)
        sdpa = torch.matmul(sdpa, self.V)


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
