import torch
import math
from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,betas=(0.9,0.95),weight_decay=1e-2,eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay rate: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr,betas=betas,weight_decay=weight_decay,eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # 初始化 m,v
                if "momentum_buffer_m" not in state:
                    state["momentum_buffer_m"] = torch.zeros_like(p.data,memory_format=torch.preserve_format)
                if "variance_buffer_v" not in state:
                    state["variance_buffer_v"] = torch.zeros_like(p.data,memory_format=torch.preserve_format)
                # 获取迭代次数 t (从 1 开始)
                t = state.get("t", 1)
                m = state["momentum_buffer_m"]
                v = state["variance_buffer_v"]
                grad = p.grad.data
                # 原地修改
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * (math.sqrt(1-beta2**t) / (1-beta1**t))
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                # 递增迭代次数
                state["t"] = t + 1
        return loss

