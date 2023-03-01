import numpy as np
import torch
import torch.nn as nn

from transformer.layers import EncoderLayer


class Encoder(nn.Module):
    """a encoder model with self attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v):
        """create multiple computation blocks"""
        super().__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head, d_k, d_v) for _ in range(n_layers)])

    def forward(self, enc_output, comm_mask, return_attns=False):
        """use self attention to merge messages"""
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, comm_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class PositionalEncoding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, d_hid, n_position=200):
        """create table"""
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """encode unique agent id """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    """a sequence to sequence model with attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position):
        """initialization"""
        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_hidden=d_hidden,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v)

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

    def forward(self, encoder_input, comm_mask):
        """run encoder"""
        encoder_input = self.position_enc(encoder_input)

        enc_output, *_ = self.encoder(encoder_input, comm_mask)

        return enc_output
