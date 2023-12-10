import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from LoRA.LoraEmbedding import LoraEmbedding
from Model.DBlock import DBlock
from Model.LayerNorm import LayerNorm
from transformers import AutoConfig


class MTGPT(nn.Module):
    """
    Implementation of a modified GPT (Generative Pre-trained Transformer) model.

    Args:
        config: Model configuration containing parameters such as vocab_size, block_size, etc.

    Attributes:
        transformer: ModuleDict containing the components of the GPT model (Embeddings, Dropout, Blocks, LayerNorm).
        lm_head: Linear layer for language modeling prediction.
        config: Model configuration.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=LoraEmbedding(num_embeddings=config.vocab_size,  # Embeddings
                              embedding_dim=config.n_embd),
            wpe=LoraEmbedding(num_embeddings=config.block_size,  # positional encodings
                              embedding_dim=config.n_embd),
            drop=nn.Dropout(config.drop_rate),
            h=nn.ModuleList([DBlock(config) for _ in range(config.n_layer)]),  # Block X N
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weights__)
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def __init_weights__(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pad_id=-1):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Max sequence length is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_id)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, Config, tokenizer, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print(f"New: vocab_size=50264, block_size=1024, bias=True")
        config_args['vocab_size'] = 50264
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = Config(**config_args)
        model = MTGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        sd_keys = [k for k in sd_keys if '.lora' not in k.lower()]

        # model configuration to which we add the special tokens
        pre_config = AutoConfig.from_pretrained('gpt2',
                                                bos_token_id=tokenizer.bos_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                output_hidden_states=False)

        # we load the pre-trained model with custom settings
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2', config=pre_config)

        # to ensure better performance: increase pad to multiple of 8
        pad_mul = 8  # in our case we have added less than 8

        # model embedding resizing
        model_hf.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_mul)

        # new tok embeds ~ N(old tok embeds)
        params_hf = model_hf.state_dict()
        old_embeds = params_hf['transformer.wte.weight']
        true_embeds = old_embeds[:-pad_mul, :]
        # build the normal distributor
        avg = torch.mean(true_embeds, dim=0)
        dof = true_embeds.size()[0] - 1
        cov = ((true_embeds - avg).T @ (true_embeds - avg)) / dof
        dist = torch.distributions.multivariate_normal.MultivariateNormal(avg, covariance_matrix=1e-5 * cov)
        # create tok_embeds as samples from this distribution
        new_embeds = torch.stack(tuple((dist.sample() for _ in range(pad_mul))), dim=0)
        old_embeds[-pad_mul:, :] = new_embeds
        params_hf['transformer.wte.weight'][-pad_mul:, :] = new_embeds
        model_hf.load_state_dict(params_hf)

        # pre-trained model params
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
