import math
import torch.nn as nn

"""
TODO: later in xalah

LoraEmbedding and LoraLinear have common attributes and methods,
making it ideal to create a shared superclass for them.
------------------------------------------------------------------------
I'm keeping it simple for now, just for the sake of simplicity.
I love the phrase: 'don't be a hero when you are learning' :)
"""


class LoRABase(nn.Module):
    """Warning: 0 < rank <<< min(in_dim, out_dim), LoRA is disabled."""

    def __init__(self, in_dim, out_dim, rank, lora_alpha, lora_dropout=0.0):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False
        # dropout OR identity f(x) = x
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x

        if rank > 0:
            # Y = Llm_w*X + lora_w*X = W*X + A*B*X
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_dim)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_dim, rank)))
            self.scale = self.lora_alpha / self.rank  # kind of a hyperparameter

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)  # init with [0,0, ...,0]
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))  # regularization technique
