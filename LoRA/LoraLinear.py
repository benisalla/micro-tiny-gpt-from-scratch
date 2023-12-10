import torch.nn as nn
import math


class LoraLinear(nn.Linear):
    """ Warning: 0 < rank <<< min(in_dim, out_dim), LoRA is disabled. """

    def __init__(self, in_dim, out_dim, rank=0, lora_alpha=1, lora_dropout=0.0, **kwargs):
        super(LoraLinear, self).__init__(in_dim, out_dim, **kwargs)

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_merged = False

        # dropout OR identity f(x) = x
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x

        if rank > 0:  # Y = Llm_w*X + lora_w*X = W*X + A*B*X
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_dim)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_dim, rank)))
            self.scale = self.lora_alpha / self.rank  # kind of a hyperparameter

        self.reset_parameters()

        # Freeze our LLM weights
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)  # init with [0,0, ...,0]
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))  # regularization technique

    def merge(self):
        if self.rank > 0 and not self.lora_merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scale
            self.lora_merged = True

    def forward(self, inputs):
        llm_w = super().forward(inputs)  # old embeds (none trainable weights)

        if self.rank <= 0 or self.lora_merged:
            return llm_w  # merged is like calling the optimizer

        lora_w = (self.lora_dropout(inputs) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scale
        return llm_w + lora_w
