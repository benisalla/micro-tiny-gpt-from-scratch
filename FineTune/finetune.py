import time

from Data.BatchLoader import BatchLoader
from Utils.Config import TConfig
from Utils.Initializer import Initializer
from Utils.Utils import save_checkpoints


# initializing
model, tokenizer, optimizer, scaler, model_args, device, ctx, iter_num, best_val_loss = Initializer()

# create dataloader for both train and val datasets
train_loader = BatchLoader(TConfig.data_dir, "train", tokenizer, TConfig.toks_per_batch)
val_loader = BatchLoader(TConfig.data_dir, "val", tokenizer, TConfig.toks_per_batch)

# hyper params
learning_rate = TConfig.learning_rate
num_steps = TConfig.num_steps
log_interval = TConfig.log_interval

# start training
t0 = time.time()
local_iter_num = 0
model =  model.to(device)

while True:
    train_loader.create_batches()
    val_loader.create_batches()

    # fetch next batch
    x_batch, y_batch = next(train_loader)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

#     iter_num, best_val_loss = evaluate_model(iter_num,
#                                              model,
#                                              optimizer,
#                                              model_args,
#                                              eval_interval,
#                                              best_val_loss,
#                                              save_ckpt_path,
#                                              save_checkpoint)

    for micro_step in range(num_steps):
        with ctx:
            logits , loss = model(x_batch, y_batch)
            loss = loss / num_steps

        x_batch, y_batch = next(train_loader)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * num_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num % 100 == 0:
        save_checkpoints(model, optimizer, model_args, iter_num, best_val_loss)

    if iter_num > TConfig.max_iters:
        break
