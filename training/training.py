# """Training."""
import numpy as np
import torch
from torch import nn
import sys
sys.path.append('..\\..')
from Transformers.model.transformer import Transformer, Decoder, LonelyDecoder
from torch.nn import functional as F
# # import Transformers.model.transformer.Transformer

text = open("""..\\Datasets\\tiny_shakespeare.txt""",
            'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
print(text[:250])

# unique characters in the file
vocab = sorted(set(text+"@"+"#")) # @ will be the initial character
                                  # and # the final character. They
                                  # are not in the text.
print('{} unique characters'.format(len(vocab)))

# Lookup tables
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print("char2idx")
print(char2idx)
print("idx2char")
print(idx2char)
print("text_as_int")
print(text_as_int)
print ('{} ---- characters mapped to int ---- >{}'.format(repr(text[:13]), text_as_int[:13]))

# Create training examples:
seq_length = 128
examples_per_epoch = len(text)//(seq_length)

int_text_tensor = torch.tensor(text_as_int)
chunks = torch.chunk(int_text_tensor, examples_per_epoch, 0)
print(int_text_tensor)

examples = [chunk[:-1] for chunk in chunks]
targets = [chunk[1:] for chunk in chunks]
print(f"""There are {len(examples)} chunks of 128 characters available for the
network training.""")



model_parameters = {
    "dim_model": 256,
    "vocabulary_size": 67,
    "batch_size": 64,
    "encoder": {
        "nb_layers": 1,
        "dim_model": 256,
        "multihead": {
            "attention": {
                "dim_model": 256,
                "dim_key": 128,
                "dim_value": 128
                },
            "nb_heads": 2
            },
        "feedforward": {
            "dim_feedforward": 256
            }
        },
    "decoder": {
        "nb_layers": 1,
        "vocabulary_size": 67,
        "dim_model": 256,
        "multihead": {
            "attention": {
                "dim_model": 256,
                "dim_key": 128,
                "dim_value": 128,
            },
            "nb_heads": 2
        },
        "feedforward": {
            "dim_feedforward": 256
        }
    }
}

# transformer = Transformer(model_parameters)
# x = torch.randn(10, 128)
# lastOutput = torch.randn(3, 128)
# transformer(x, lastOutput)

decoder = LonelyDecoder(model_parameters)
x = torch.randn(10, model_parameters["vocabulary_size"])
lastOutput = torch.randn(3, model_parameters["vocabulary_size"])
decoder(x)


one_hot_examples = F.one_hot(torch.stack(examples[:-1]).long(),
                             model_parameters["vocabulary_size"]).float()
one_hot_targets = F.one_hot(torch.stack(targets[:-1]).long(), model_parameters["vocabulary_size"]).float()

decoder(one_hot_examples[0])
one_hot_examples[0]

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(decoder(one_hot_examples[0]), one_hot_targets[0])
loss.backward()
optimizer = torch.optim.SGD(decoder.parameters())

from torch.utils.data import DataLoader, Dataset

data = torch.stack((one_hot_examples, one_hot_targets), dim=0)
class customDataset(Dataset):
    def __init__(self, data):
        self.data  = data
    def __len__(self):
        return data.shape[1]
    def __getitem__(self, idx):
        return data[0,idx], data[1, idx]

dataset = customDataset(data)

train_dataloader = DataLoader(customDataset(data[:,0:10]), batch_size=1, shuffle=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    """Train loop. Taken from pytorch tutorial."""
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch
    # normalization and dropout layers Unnecessary in this situation
    # but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = char2idx["@"]
        pred = F.one_hot(torch.Tensor([pred]).long(),
                         model_parameters["vocabulary_size"]).float()
        print(f"Initial shape: {pred.shape}")
        for i in range(seq_length-1):
            print(f"Shape of the prediction: {model(pred).shape}")
            pred = torch.cat((pred, model(pred)))
            print(f"Shape at iteration {i}: {pred.shape}")
            print("Prediction:")
            print(''.join([idx2char[i] for i in torch.max(pred,
                                                          1)[1].tolist()]))
        loss = loss_fn(pred[1:], y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_loop(train_dataloader, decoder, loss_fn, optimizer)



# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")
# #print(transformer)
