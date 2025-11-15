import torch
import torch.nn as nn
from pydantic.experimental.pipeline import transform

from Modules.rope import RotaryPositionalEmbedding
from Modules.multi_self_attention import MutiHeadSelfAttention
from Modules.rms_norm import RMSNorm
from Modules.linear import Linear
from Modules.swigu import SwiGLU
from Modules.embedding import Embedding
from Modules.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 context_length,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 rope:RotaryPositionalEmbedding = None):
        super().__init__()

        # token_embedding
        self.token_embeddings = Embedding(vocab_size,d_model)

        # transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
                max_seq_len=context_length
            )
            self.layers.append(block)

        # norm
        self.ln_final = RMSNorm(d_model)

        # linear
        self.lm_head = Linear(d_model,vocab_size)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x 形状: (batch_size, seq_len) -- 传入的是 token ID

        # 1. token_embeddings
        # 变为: (batch_size, seq_len, d_model)
        x = self.token_embeddings(x)

        # 2. layers (N 个 TransformerBlock)
        # 形状保持: (batch_size, seq_len, d_model)
        for block in self.layers:
            x = block(x)

        # 3. ln_final (最终归一化)
        # 形状保持: (batch_size, seq_len, d_model)
        x = self.ln_final(x)

        # 4. lm_head (投影回词汇空间)
        # 变为: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits