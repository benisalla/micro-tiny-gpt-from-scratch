import os
import time
import numpy as np
import torch
from Utils.Config import TConfig
from Utils.Initializer import Initializer
from Utils.Utils import save_checkpoints, get_lr, evaluate_model, get_batch

# connect to dataset
train_data = np.memmap(f"{TConfig.data_dir}/val.bin", dtype=np.uint16, mode='r')
val_data = np.memmap(f"{TConfig.data_dir}/val.bin", dtype=np.uint16, mode='r')

# initializing
model, tokenizer, optimizer, scaler, model_args, device, ctx, iter_num, best_val_loss = Initializer()

# hyper params
block_size = TConfig.block_size
batch_size = TConfig.batch_size
learning_rate = TConfig.learning_rate
decay_lr = TConfig.decay_lr
num_steps = TConfig.num_steps
grad_clip = TConfig.grad_clip
log_interval = TConfig.log_interval

# start training
X, Y = get_batch(train_data, device)
t0 = time.time()
local_iter_num = 0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    iter_num, best_val_loss = evaluate_model(iter_num, model, train_data, val_data, device, ctx, best_val_loss)

    for micro_step in range(num_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / num_steps

        X, Y = get_batch(train_data, device)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * num_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num % 10 == 0:
        save_checkpoints(model, optimizer, model_args, iter_num, best_val_loss)

    if iter_num > TConfig.max_iters:
        break
