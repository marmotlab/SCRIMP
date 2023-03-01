import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature):
        """initialization"""
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, comm_mask):
        """ run multiple independent attention heads in parallel"""
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        comm_mask = torch.unsqueeze(comm_mask, 1)
        attn = attn.masked_fill(comm_mask == 0, -float("inf"))  # if mask==0,the input value will =-1e6
        # then the attention score will around 0
        attn = F.softmax(attn, dim=-1)  # attention score
        output = torch.matmul(attn, v)
        return output, attn
