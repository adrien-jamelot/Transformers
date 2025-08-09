"""Residual connection."""


def addAndNorm(x, blockOutput, norm):
    return norm(x + blockOutput)
