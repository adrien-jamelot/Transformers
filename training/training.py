# """Training."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


tiny_shakespeare =open('/content/drive/MyDrive/Transformers/data/tiny_shakespeare.txt',
     'rb').read().decode(encoding='utf-8')
text = tiny_shakespeare
print('Length of text: {} characters'.format(len(text)))
print(text[:250])

# unique characters in the file
vocab = sorted(set(text+"@"+"#")) # @ will be the initial character
                                  # and # the final character. They
                                  # are not in the text.
print('{} unique characters'.format(len(vocab)))
