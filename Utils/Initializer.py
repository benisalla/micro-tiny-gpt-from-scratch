import os
import torch
from contextlib import nullcontext
from Model.MTGPT import MTGPT
from Utils.Config import TConfig
from Utils.Utils import create_tokenizer


def Initializer():
    torch.manual_seed(2000)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)

    # init Tokenizer
    tokenizer = create_tokenizer("gpt2")

    # some local variables
    vocab_size = len(tokenizer.get_vocab())
    init_from = TConfig.init_from
    iter_num = 0
    best_val_loss = 1e9

    # build Model params from TConfig
    model_args = dict(n_layer=TConfig.n_layer,
                      n_head=TConfig.n_head,
                      n_embd=TConfig.n_embd,
                      block_size=TConfig.block_size,
                      bias=TConfig.bias,
                      vocab_size=vocab_size,
                      drop_rate=TConfig.drop_rate)

    if init_from == 'scratch':
        model_args['vocab_size'] = vocab_size
        config = TConfig(**model_args)
        model = MTGPT(config)

    elif init_from == 'resume':
        checkpoint = None
        if os.path.exists(TConfig.save_ckpt_path):
            checkpoint = torch.load(TConfig.save_ckpt_path, map_location=device)
        else:
            checkpoint = torch.load(TConfig.load_ckpt_path, map_location=device)

        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        config = TConfig(**model_args)
        model = MTGPT(config)
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        for key, value in list(state_dict.items()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print("checkpoints loaded successfully :)")
    else:
        print("Oups, There is no such option !!")

    # Crop block size of the original model
    if TConfig.block_size < model.config.block_size:
        model.crop_block_size(TConfig.block_size)
        model_args['block_size'] = TConfig.block_size
    model.to(device)

    # initialize a GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    optimizer = model.configure_optimizers(TConfig.weight_decay,
                                           TConfig.learning_rate,
                                           (TConfig.beta1, TConfig.beta2),
                                           device)

    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    # compile the model for to more efficient
    if TConfig.compile:
        model = torch.compile(model)

    return model, tokenizer, optimizer, scaler, model_args, device, ctx, iter_num, best_val_loss
