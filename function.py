import numpy as np
import torch
from einops import einsum,rearrange
import math
from jaxtyping import Bool, Float, Int, Array
from torch import Tensor
import numpy.typing as npt
from Modules.transformer_lm import TransformerLM

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
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    
    start_indices = np.random.randint(0, n - context_length, size=(batch_size,))
    offsets = np.arange(context_length + 1)
    indices = start_indices[:, None] + offsets

    chunk = dataset[indices]

    chunk_tensor = torch.from_numpy(chunk).to(device, non_blocking=True)
    if chunk_tensor.dtype != torch.int64:
        chunk_tensor = chunk_tensor.to(torch.int64)
    
    x = chunk_tensor[:, :-1]
    y = chunk_tensor[:, 1:]
    return x, y

def decode(
        model:TransformerLM,
        prompt: Int[Tensor, "B T"],
        maximum_num:Int,
        p:Float,
        temperature: float = 1.0,
        eos_token_id: int = 256
) -> Int[Tensor, " ..."]:
    """
    generate content with given prompt
    """
    # 主循环
    # 将prompt喂给模型预测，获取logits
    # 使用温度采样和top-p采样获取预测结果的token
    # 将生成的token加入prompt继续训练直到遇到eof或者达到最大生成数量
    model.eval()
    iter_num = 0
    finished = torch.zeros(prompt.shape[0],dtype=torch.bool,device=prompt.device)
    with torch.no_grad():
        while iter_num < maximum_num and not finished.all():
            # 如果当前prompt超过最大上下文长度就截断
            prompt_truncate = prompt if prompt.shape[1] <= model.context_length else prompt[:,-model.context_length:]
            logits = model(prompt_truncate)
            predict_token_logits = logits[:,-1,:] / temperature
            distribution = softmax(predict_token_logits,dim=-1)
            # 按行排序，累计求和
            sorted_distribution,sorted_indices = torch.sort(distribution,descending=True)
            cumulative_distr = torch.cumsum(sorted_distribution,dim=-1)
            indices_to_remove = cumulative_distr > p
            # 保留刚好>p的indice(右移一位)
            indices_to_remove[...,1:] = indices_to_remove[...,:-1].clone()
            indices_to_remove[...,0] = 0
            # 重新排序
            mask_in_original_order = torch.zeros_like(distribution, dtype=torch.bool)
            mask_in_original_order.scatter_(dim=1, index=sorted_indices, src=indices_to_remove)
            predict_token_logits[mask_in_original_order] = float('-inf')
            new_distr = softmax(predict_token_logits,dim=-1)
            next_token = torch.multinomial(new_distr,1)

            # 1. 获取当前采样的 token 数值 (Shape: [B])
            next_token_vals = next_token.squeeze(-1)
            
            # 2. 更新 finished 状态
            # 如果某行这次生成了 EOS，就把它的 finished 设为 True
            # 使用逻辑或 (|)，一旦变成 True 就永远是 True
            finished = finished | (next_token_vals == eos_token_id)
            
            # 3. 处理“由于已经结束”而产生的无效 token
            # 如果某一行在【上一轮】就已经 finished 了，
            # 那么这一轮采样的结果是不重要的，我们强制把它覆盖为 EOS (作为 Padding)
            # 这样输出的 Tensor 后面就是 neat 的 EOS, EOS, EOS...
            # 注意：这里要把 finished 扩充维度变成 [B, 1] 才能和 next_token 操作
            next_token = torch.where(
                finished.unsqueeze(-1),      # 条件: 如果已经结束
                torch.tensor(eos_token_id, device=prompt.device), # True: 填 EOS
                next_token                   # False: 填原本采样结果
            )

            prompt = torch.cat((prompt,next_token),dim=1)
            iter_num += 1
    return prompt

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