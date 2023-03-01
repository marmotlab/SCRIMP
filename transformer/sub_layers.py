import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """multi-head self attention module"""

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        """initialization"""
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, comm_mask):
        """calculate multi-head attention"""
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # calculate attention
        q, attn = self.attention(q, k, v, comm_mask)
        # combine the last two dimensions to concatenate all the heads together
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid):
        """Initialization"""
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        """run a ff layer"""
        x = self.w_2(F.relu(self.w_1(x)))
        return x


class GatingMechanism(nn.Module):
    """a GRU cell"""

    def __init__(self, d_model, bg=2):
        """Initialization"""
        super(GatingMechanism, self).__init__()
        self.Wr = nn.Linear(d_model, d_model)
        self.Ur = nn.Linear(d_model, d_model)
        self.Wz = nn.Linear(d_model, d_model)
        self.Uz = nn.Linear(d_model, d_model)
        self.Wg = nn.Linear(d_model, d_model)
        self.Ug = nn.Linear(d_model, d_model)
        self.bg = torch.nn.Parameter(torch.full([d_model], bg, dtype=torch.float32))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):  # x is residual, y is input
        """run a GRU in the place of residual connection"""
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g
