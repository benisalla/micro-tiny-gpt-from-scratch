from torch import nn


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
        self.fc_ln = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj_ln = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = self.gelu(self.fc_ln(x))
        return self.dropout(self.proj_ln(x))
