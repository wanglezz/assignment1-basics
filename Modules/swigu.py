import torch
import torch.nn as nn
import math
from Modules import linear
from Modules.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()

        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        w1x = self.w1.forward(x)
        silu_x = w1x * torch.sigmoid(w1x) #SiLU(W1x)
        w3_x = self.w3.forward(x) # W3x
        return self.w2.forward(silu_x * w3_x) # w2(SiLU(w1x) * W3x)
