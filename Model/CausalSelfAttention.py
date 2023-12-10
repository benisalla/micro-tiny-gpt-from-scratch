import math
import torch
from torch import nn
from torch.nn import functional as F

from LoRA.LoraLinear import LoraLinear


class CausalSelfAttention(nn.Module):
    """
        Causal Self-Attention Module
        ( why causal? well because attention mechanism is a causal operation not a correlation )

        Tensor          Type            Shape
        ===========================================================================
        input           long            (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0; "n_embd % n_head should be 0."
        """   
            dropout ~= lora_dropout since Pre-Trained W is frozen
            we can either introduce new dropout (lora_dropout) or use dropout
        """
        self.c_attn = LoraLinear(in_dim=config.n_embd,
                                 out_dim=3 * config.n_embd,
                                 bias=config.bias,
                                 rank=config.rank,
                                 lora_alpha=config.lora_alpha,
                                 lora_drop_rate=config.lora_dropout)
        self.c_proj = LoraLinear(in_dim=config.n_embd,
                                 out_dim=config.n_embd,
                                 bias=config.bias,
                                 rank=config.rank,
                                 lora_alpha=config.lora_alpha,
                                 lora_drop_rate=config.lora_dropout)

        self.atten_drop = nn.Dropout(config.drop_rate)
        self.res_drop = nn.Dropout(config.drop_rate)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.drop_rate = config.drop_rate
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Works only on PyTorch version 2.0 or higher.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.mh_atten_ln(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.drop_rate if self.training else 0,
                                                                 is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.atten_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.res_drop(self.proj_ln(y))
        return y
