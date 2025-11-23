import numpy as np
import torch
from einops import einsum,rearrange
import math
from jaxtyping import Bool, Float, Int, Array
from torch import Tensor
import numpy.typing as npt

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon (M1/M2/M3...)
    else:
        return torch.device("cpu")   # 只有 CPU

def softmax(x:torch.Tensor,dim:int) -> torch.Tensor:
    max_val =  torch.max(x,dim=dim,keepdim=True).values
    stable_x = x - max_val
    numerator = torch.exp(stable_x)
    denominator = torch.sum(numerator,dim=dim,keepdim=True)
    output = numerator / denominator
    return output

def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    attention_scores = einsum(
        Q,
        K,
        "... seq_len_q d_k,... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==False,-1e9)

    attention_weights = softmax(attention_scores,dim=-1)
    attention = einsum(
        attention_weights,
        V,
        "... seq_len_q seq_len_k,... seq_len_k d_v -> ... seq_len_q d_v"
    )
    return attention

def cross_entropy(
        logits: Float[Tensor, " ... vocab_size"],
        targets: Float[Tensor, " ... "]
):
    # 展平张量
    logits_flat = rearrange(logits,"... vocab_size -> (...) vocab_size")
    targets_flat = rearrange(targets,"... -> (...)")

    max_logits = logits_flat.max(dim=-1,keepdim=True).values
    stable_logits = logits_flat - max_logits
    log_sum_exp = max_logits.squeeze(-1) +torch.log(torch.exp(stable_logits).sum(dim=-1))

    items = targets_flat.shape[0]
    correct_logits = logits_flat[torch.arange(items,device=logits.device),targets_flat]

    loss_per_token = log_sum_exp - correct_logits

    return loss_per_token.mean()

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # warm up
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    return min_learning_rate

def gradient_clipping(parameters, max_l2_norm: float,eps=1e-6):
    norm_sum = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            norm_sum += param_norm**2
    norm_sum = math.sqrt(norm_sum)
    if norm_sum > max_l2_norm:
        clip_coef = max_l2_norm / (norm_sum + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

def data_loader(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str = 'mps'
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    index = torch.randint(n-context_length,(batch_size,))

    x = torch.stack([torch.from_numpy(dataset[i:i+context_length].astype(np.int64))for i in index])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1].astype(np.int64))for i in index])

    if device == "mps":
        x,y = x.to(device),y.to(device)
    return x,y

def save_checkpoint(model, optimizer, iteration, out):
    check_point = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(check_point,out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


if __name__ == "__main__":
    print(get_device())