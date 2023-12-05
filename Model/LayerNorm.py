import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):

    """
        Layer Normalization module. (different than batch normalization and better in seq problems)

        Tensor          Type            Shape
        ===========================================================================
        input           long            (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
