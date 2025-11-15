import torch
from einops import einsum,rearrange
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor


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
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.exp(stable_logits).sum(dim=-1))

    items = targets_flat.shape[0]
    correct_logits = logits_flat[torch.arange(items,device=logits.device),targets_flat]

    loss_per_token = log_sum_exp - correct_logits

    return loss_per_token.mean()