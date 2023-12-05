import math
import torch
from transformers import GPT2Tokenizer
from Utils.Config import TConfig
import numpy as np


def get_batch(data, device):
    ix = torch.randint(len(data) - TConfig.block_size, (TConfig.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + TConfig.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + TConfig.block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model,
                  device,
                  train_data,
                  val_data,
                  ctx=None):
    eval_iters = TConfig.eval_iters
    out = {}
    model.eval()
    data_dict = {"train": train_data, "val": val_data}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dict[split], device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(iter):
    min_lr = TConfig.min_lr
    lr = TConfig.learning_rate
    w = TConfig.warmup_iters
    lr_di = TConfig.lr_decay_iters
    if iter < w:
        return lr * iter / w
    if iter > lr_di:
        return min_lr
    decay_ratio = (iter - w) / (lr_di - w)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def save_checkpoints(model,
                     optimizer,
                     model_args,
                     iter_num,
                     best_val_loss):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    print("Checkpoints Saved Successfully :)")
    torch.save(checkpoint, TConfig.save_ckpt_path)


def evaluate_model(iter_num, model, train_data, val_data, device, ctx, best_val_loss):
    if iter_num % TConfig.eval_interval == 0:
        losses = estimate_loss(model, device, train_data, val_data, ctx)
        print(f"Val-Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    return iter_num, best_val_loss


def create_tokenizer(ttype):
    # Additional special tokens dictionary
    special_tokens = ['<ans>']
    spe_tokens_dict = {
        'pad_token': '<pad>',
        'bos_token': '<sost>',
        'eos_token': '<eost>',
        'additional_special_tokens': special_tokens
    }

    # Add special tokens to the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(ttype)
    tokenizer.add_special_tokens(spe_tokens_dict)

    return tokenizer
