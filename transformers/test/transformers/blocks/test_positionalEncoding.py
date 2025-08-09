from transformers.blocks.positionalEncoding import positionalEncoding
import torch


def test_positionalEncoding_with_correct_input():
    sequenceLength = 10
    embeddingDimension = 20
    x = torch.randn(sequenceLength, embeddingDimension)
    y = positionalEncoding(x)

    # Test for even values
    for pos in range(sequenceLength):
        for i in range(embeddingDimension//2):
            assert(y[pos,i] == x[pos,i] + torch.sin(torch.tensor(pos/1000**(2*i/embeddingDimension))))
            
    
