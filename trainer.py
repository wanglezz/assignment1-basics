import time
from sqlite3 import adapters

import torch
import numpy as np
import argparse,yaml,os
from Modules.bpe_tokenizer import BPETokenizer
from Modules.rope import RotaryPositionalEmbedding
from Modules.transformer_lm import TransformerLM
from Modules.adamw import AdamW
from function import data_loader, cross_entropy,gradient_clipping,save_checkpoint,load_checkpoint
from tests.adapters import run_train_bpe

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')

    # 模型超参
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--context_length', type=int, default=1024, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='model dim')
    parser.add_argument('--d_ff', type=int, default=2048, help='ff dim')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=384)

    # 训练超参
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=200, help='How often to evaluate val loss')
    parser.add_argument('--eval_iters', type=int, default=50, help='How many batches to verify')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
    parser.add_argument('--rope_theta', type=float, default=10000.0,
                        help='Base frequency for RoPE. Increase for longer context.')
    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='transformer-project')

    return parser.parse_args()

args = get_args()
# 创建输出目录
os.makedirs(args.out_dir, exist_ok=True)

def get_data(spilt):
    data_path = os.path.join(args.data_dir,f"{spilt}.bin")
    data = np.memmap(data_path,dtype=np.uint16,mode='r')
    return data_loader(data,args.batch_size,args.context_length)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for spilt in ['train','val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x,y = get_data(spilt)
            logits,targets = model(x),y
            loss = cross_entropy(logits,targets)
            losses[k] = loss.item()
        out[spilt] = losses.mean()
    model.train()
    return out

def train():
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.n_layer,
        d_model=args.d_model,
        num_heads=args.n_head,
        d_ff=args.d_ff,
        rope=RotaryPositionalEmbedding(
            args.rope_theta,
            args.d_model // args.n_heads,
            args.context_length
        )
    )
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(),lr=args.learning_rate)

    best_val_loss = float('inf')
    iter_num = 0
    t0 = time.time()

    while iter_num < args.max_iters:
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if args.use_wandb:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": args.learning_rate,
            })
        if iter_num % 1000 == 0:
            print(f"Saving checkpoint to {args.out_dir}/ckpt.pt")
            save_checkpoint(model,optimizer,iter_num,os.path.join(args.out_dir, f'{iter_num}.pt'))

        # train
        x,y = get_data('train')
        logits = model(x)
        loss = cross_entropy(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_clipping(model.parameters(),1.0)

        optimizer.step()

        if iter_num % 10 == 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000  # 毫秒
            print(f"step {iter_num} | loss {loss.item():.4f} | time {dt:.2f}ms")
            t0 = t1

        iter_num += 1