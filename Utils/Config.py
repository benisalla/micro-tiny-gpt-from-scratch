from dataclasses import dataclass


@dataclass
class TConfig:
    eval_interval: int = 20
    log_interval: int = 1
    eval_iters: int = 1 #200
    num_steps: int = 1 #200
    batch_size: int = 2
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2  # 12
    n_head: int = 4 #12
    n_embd: int = 32 #768
    drop_rate: float = 0.0
    bias: bool = True
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    compile: bool = False

    # could be "scratch", "resume", "GPT2 family"
    init_from: str = "scratch"

    # directories
    data_dir: str = "./Data"
    save_ckpt_path: str = "./Checkpoints/ckpt.pt"
    load_ckpt_path: str = "./Checkpoints/ckpt.pt"
