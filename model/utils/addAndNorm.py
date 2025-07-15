"""Residual connection."""


def addAndNorm(x, blockOutput, norm):
    """Residual connection."""
    return norm(x + blockOutput)
