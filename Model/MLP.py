from torch import nn

from LoRA.LoraLinear import LoraLinear


class MLP(nn.Module):
    """
        Multi-Layer Perceptron (MLP): computation layer.
        Non-linearity = complex mappings + feature extraction  + ... + processing data (reasoning)

        Tensor          Type            Shape
        ===========================================================================
        input           long            (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================

    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = LoraLinear(in_dim=config.n_embd,
                               out_dim=4 * config.n_embd,
                               bias=config.bias,
                               rank=config.rank,
                               lora_alpha=config.lora_alpha,
                               lora_drop_rate=config.lora_dropout)
        self.gelu = nn.GELU()
        self.c_proj = LoraLinear(in_dim=4 * config.n_embd,
                                 out_dim=config.n_embd,
                                 bias=config.bias,
                                 rank=config.rank,
                                 lora_alpha=config.lora_alpha,
                                 lora_drop_rate=config.lora_dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.gelu(self.fc_ln(x))
        return self.dropout(self.proj_ln(x))
