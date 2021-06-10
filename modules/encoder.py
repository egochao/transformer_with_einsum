import torch
from torch import nn
from .attention import CustomMultiHeadAttention
from .blocks import PositionwiseFeedForward, CustomLayerNorm


class CustomEncoderLayer(nn.Module):
    def __init__(self, dim, n_head, ffn_hidden=None, dropout=0.0):
        super(CustomEncoderLayer, self).__init__()
        self.multiheadatt = CustomMultiHeadAttention(dim, n_head)
        self.norm1 = CustomLayerNorm(dim)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForward(dim, ffn_hidden)
        self.norm2 = CustomLayerNorm(dim)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # Compute attention
        _x = x
        x = self.multiheadatt(x, x, x, mask=mask)

        # add norm

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # feed forward
        _x = x
        x = self.ffn(x)

        # add norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x



