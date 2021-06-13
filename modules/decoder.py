import torch
from torch import nn
from .attention import CustomMultiHeadAttention
from .blocks import PositionwiseFeedForward, CustomLayerNorm
from .position_layers import PositionEncoding


class CustomDecoderLayer(nn.Module):
    def __init__(self, dim, n_head, ffn_hidden, dropout):
        """
        Decoder block
        :param dim: Embedding dimension
        :param n_head: Number of head in multi head attention
        :param ffn_hidden: Number of hidden nodes of feed forward layer
        :param dropout: Dropout rate of the block
        """
        super(CustomDecoderLayer, self).__init__()

        self.multiheadatt_forcing = CustomMultiHeadAttention(dim, n_head)
        self.norm1 = CustomLayerNorm(dim)
        self.dropout1 = nn.Dropout(p=dropout)

        self.multiheadatt_encode = CustomMultiHeadAttention(dim, n_head)
        self.norm2 = CustomLayerNorm(dim)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForward(dim, ffn_hidden, dropout=dropout)
        self.norm3 = CustomLayerNorm(dim)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1 compute self attention of label(with masking)
        _x = dec
        x = self.multiheadatt_forcing(dec, dec, dec, mask=trg_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.multiheadatt_encode(x, enc, enc, mask=src_mask)
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        _x = x
        x = self.ffn(x)
        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x


class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, dim, ffn_hidden, n_head, n_layers, dropout=0.1):
        """
        Decoder (n x decoder layer)
        :param vocab_size: Output vocab size for embedding
        :param max_len: Maximum length of position embedding
        :param dim: Embedding dimension
        :param ffn_hidden: Number of hidden nodes of feed forward layer
        :param n_head: Number of head in multi head attention
        :param n_layers: Number of repeated encoder layers
        :param dropout: Dropout rate of encoder
        """
        super(CustomDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=1)
        self.position = PositionEncoding(dim, dropout=dropout, max_len=max_len)

        self.layers = nn.ModuleList([CustomDecoderLayer(dim,
                                                        n_head,
                                                        ffn_hidden,
                                                        dropout)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.embed(trg)
        trg = self.position(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output = self.linear(trg)

        return output




