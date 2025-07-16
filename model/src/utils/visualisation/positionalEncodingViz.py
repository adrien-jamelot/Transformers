"""Checking that the encoded position looks like something."""
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('..')
from PositionalEncoding import positionalEncoding

x = torch.ones((32, 128))  # batch_size, sequence_size, dimension
posEncoding = positionalEncoding(x, dim_model=128)
print(posEncoding[:, 2])
for i in range(32):
    plt.plot(posEncoding[:, i])
plt.show()
