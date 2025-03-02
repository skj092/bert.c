import torch
import numpy as np
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def check_layer_normalization(tensor, atol=1e-4):
    """
    Checks whether a tensor is layer normalized.

    Args:
        tensor (torch.Tensor): The input tensor (assumed shape: [batch, seq_len, embedding_dim]).
        atol (float): Absolute tolerance for floating-point comparisons.

    Returns:
        bool: True if the tensor is approximately layer normalized, False otherwise.
    """
    # Compute mean and variance along the last dimension (embedding_dim)
    means = tensor.mean(dim=-1)  # Shape: [batch, seq_len]
    variances = tensor.var(dim=-1, unbiased=False)  # PyTorch variance defaults to an unbiased estimate

    print("Mean per row:\n", means)
    print("Variance per row:\n", variances)

    mean_check = torch.allclose(means, torch.zeros_like(means), atol=atol)
    variance_check = torch.allclose(variances, torch.ones_like(variances), atol=atol)

    print("Mean check:", mean_check)
    print("Variance check:", variance_check)

    return mean_check and variance_check


batch, sentence_length, embedding_dim = 2, 4, 8
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim, eps=1e-5)  # Default PyTorch LayerNorm uses small eps

# Activate module
out = layer_norm(embedding)

# Debug output
print("LayerNorm Output:\n", out)

# Check LayerNorm
print("Is Layer Normalized?", check_layer_normalization(out))
