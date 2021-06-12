import torch
from torch import nn
from .attention import CustomMultiHeadAttention
from .blocks import PositionwiseFeedForward, CustomLayerNorm
from .position_layers import PositionEncoding


class CustomEncoderLayer(nn.Module):
    def __init__(self, dim, n_head, ffn_hidden=None, dropout=0.0):
        """
        Encoder block
        :param dim: Embedding dimension
        :param n_head: Number of head in multi head attention
        :param ffn_hidden: Number of hidden nodes of feed forward layer
        :param dropout: Dropout rate in the block
        """
        super(CustomEncoderLayer, self).__init__()
        self.multiheadatt = CustomMultiHeadAttention(dim, n_head)
        self.norm1 = CustomLayerNorm(dim)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForward(dim, ffn_hidden, dropout=dropout)
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


class CustomEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, dim, ffn_hidden, n_head, n_layers, dropout=0.1):
        """
        Encoder (n x encode layer)
        :param vocab_size: Input vocab size for embedding
        :param max_len: Maximum length of position embedding
        :param dim: Embedding dimension
        :param ffn_hidden: Number of hidden nodes of feed forward layer
        :param n_head: Number of head in multi head attention
        :param n_layers: Number of repeated encoder layers
        :param dropout: Dropout rate of encoder
        """
        super(CustomEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=1)
        self.position = PositionEncoding(dim, dropout=dropout, max_len=max_len)

        self.layers = nn.ModuleList([CustomEncoderLayer(dim,
                                                        n_head,
                                                        ffn_hidden,
                                                        dropout)
                                     for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.position(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
