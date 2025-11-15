import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        indices = torch.arange(0,d_k,2,device=device).float()
        exp = indices / d_k
        inv_freq = 1 / (theta ** exp)
        pos = torch.arange(max_seq_len,device=device)
        angles = torch.outer(pos,inv_freq)
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        self.register_buffer("cos_cached",cos_values,persistent=False)
        self.register_buffer("sin_cached",sin_values,persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[...,::2]
        x_odd = x[...,1::2]

        x_rotate_even = cos * x_even - sin * x_odd
        x_rotate_odd = sin * x_even + cos * x_odd

        target_x = torch.empty_like(x)
        target_x[...,::2] = x_rotate_even
        target_x[...,1::2] = x_rotate_odd

        return target_x