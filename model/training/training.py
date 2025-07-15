# """Training."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import Decoder

tiny_shakespeare = (
    open("../data/tiny_shakespeare.txt", "rb").read().decode(encoding="utf-8")
)
text = tiny_shakespeare

print("Length of text: {} characters".format(len(text)))
print("First 250 characters: " + text[:250])

vocab = sorted(set(text + "@" + "#"))
print("{} unique characters".format(len(vocab)))
# @ will be the initial character and # the final character. They are not in the text.

# Lookup tables
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print("char2idx")
print(char2idx)
print("idx2char")
print(idx2char)
print("text_as_int")
print(text_as_int)
print(
    "{} ---- characters mapped to int ---- >{}".format(
        repr(text[:13]), text_as_int[:13]
    )
)

# Create training examples:
seq_length = 16
examples_per_epoch = len(text) // (seq_length)

int_text_tensor = torch.tensor(text_as_int)
chunks = torch.chunk(int_text_tensor, examples_per_epoch, 0)
print(int_text_tensor)

examples = [chunk[:-1] for chunk in chunks]
targets = [chunk[1:] for chunk in chunks]
print(f"""There are {len(examples)} chunks of {seq_length} characters available for the
network training.""")


model_parameters = {
    "nb_layers": 1,
    "vocabulary_size": 67,
    "dim_model": 512,
    "dim_key": 64,
    "dim_value": 64,
    "nb_heads": 8,  # must be so that nb_heads * dim_value = dim_model else the concat won't make sense
}
decoder = Decoder(model_parameters)
optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-3, betas=(0.9, 0.98))
dataset = customDataset(one_hot_examples, targets)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
run_n_epochs(5, train_dataloader, decoder, loss_fn, optimizer)
