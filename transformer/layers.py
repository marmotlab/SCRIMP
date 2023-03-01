import torch.nn as nn

from transformer.sub_layers import MultiHeadAttention, PositionwiseFeedForward, GatingMechanism


class EncoderLayer(nn.Module):
    """compose with two different sub-layers"""

    def __init__(self, d_model, d_hidden, n_head, d_k, d_v):
        """define one computation block"""
        super(EncoderLayer, self).__init__()
        self.gate1 = GatingMechanism(d_model)
        self.gate2 = GatingMechanism(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_input):
        """run a computation block"""
        enc_output = self.norm1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output)
        enc_output_1 = self.gate1(enc_input, enc_output)
        enc_output = self.pos_ffn(self.norm2(enc_output_1))
        enc_output = self.gate2(enc_output_1, enc_output)
        return enc_output, enc_slf_attn
