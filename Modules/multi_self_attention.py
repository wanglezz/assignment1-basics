import torch
import torch.nn as nn
from Modules import linear
from Modules.linear import Linear
from Modules.rope import RotaryPositionalEmbedding
from einops import rearrange
from function import scaled_dot_product_attention

class MutiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,nums_head:int,max_seq_len: int = 2048,rope:RotaryPositionalEmbedding=None):
        super().__init__()
        assert d_model % nums_head == 0
        self.nums_head = nums_head
        self.d_k = d_model // nums_head
        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        self.output_proj = Linear(d_model,d_model)
        self.rope = rope
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False
        )

    def forward(self,x:torch.Tensor,token_positions=None) -> torch.Tensor:
        B,S,_ = x.shape
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)
        Q = rearrange(Q,"b s (h d_k) -> b h s d_k",h=self.nums_head)
        K = rearrange(K, "b s (h d_k) -> b h s d_k",h=self.nums_head)
        V = rearrange(V, "b s (h d_k) -> b h s d_k",h=self.nums_head)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(S, device=x.device)
                # 确保外部传入的也是在正确的 device 上 (安全起见)
            elif token_positions.device != x.device:
                token_positions = token_positions.to(x.device)
            Q,K= self.rope(Q,token_positions),self.rope(K,token_positions)
        mask = self.mask[0:S,0:S]
        attention = scaled_dot_product_attention(Q, K, V,mask)
        output = rearrange(attention,"b h s d_k -> b s (h d_k)")
        output = self.output_proj.forward(output)

        return output