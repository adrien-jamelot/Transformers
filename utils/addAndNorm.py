"""Residual connection."""


def addAndNorm(x, block, norm):
    """Residual connection."""
    return norm(x + block(x))
