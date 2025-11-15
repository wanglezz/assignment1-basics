import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None,dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        variance = 2.0 / (in_features + out_features)
        std = math.sqrt(variance)
        lower_bound = -3 * std
        upper_bound = 3 * std
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a=lower_bound,
            b=upper_bound
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... in_features,out_features in_features -> ... out_features"
        )