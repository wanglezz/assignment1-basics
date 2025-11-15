import torch
import torch.nn as nn
from Modules.multi_self_attention import MutiHeadSelfAttention
from Modules.rms_norm import RMSNorm
from Modules.rope import RotaryPositionalEmbedding
from Modules.linear import Linear
from Modules.swigu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,rope:RotaryPositionalEmbedding =None,max_seq_len: int =2048):
        super().__init__()
        # MHA 子层
        self.ln1 = RMSNorm(d_model)
        self.attn = MutiHeadSelfAttention(
            d_model=d_model,
            nums_head=num_heads,
            max_seq_len=max_seq_len,
            rope=rope
        )

        # FFN 子层
        self.ln2 = RMSNorm( d_model)
        self.ffn = SwiGLU(d_model,d_ff)

    def forward(self,x:torch.Tensor,token_positions=None) -> torch.Tensor:
        # MHA
        x = x + self.attn.forward(self.ln1.forward(x),token_positions)

        # FFN
        x = x + self.ffn.forward(self.ln2.forward(x))

        return x