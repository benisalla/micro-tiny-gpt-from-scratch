from torch import nn

from Model.CausalSelfAttention import CausalSelfAttention
from Model.MLP import MLP
from Model.LayerNorm import LayerNorm


class DBlock(nn.Module):

    """
        Transformer Block: communication layer (att mechanism) + computation layer ( feedforward).

        Tensor          Type            Shape
        ===========================================================================
        input           long            (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn_net = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ff_mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn_net(self.ln_1(x))
        return x + self.ff_mlp(self.ln_2(x))
