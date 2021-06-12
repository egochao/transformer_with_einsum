import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class PositionEncoding(nn.Module):
    """Implement the PE function."""
    # MODE_EXPAND = 'EXPAND'
    MODE_ADD = 'ADD'
    MODE_CONCAT = 'CONCAT'

    def __init__(self, dim, dropout=0.0, max_len=5000, mode="ADD"):
        """
        Sinusoidal encoding of position, fix during training
        :param dim: Embedding dimension (equal to the input dimension in case of mode ADD)
        :param dropout: Dropout rate of final output
        :param max_len: Maximum len of input sequence length
        :param mode: Type of merging input and position encoding
        """

        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _pe = self.pe[:, :x.size(1)]
        batch_size, seq_len = x.size()[:2]
        if self.mode == self.MODE_ADD:
            x = x + _pe
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, _pe.repeat(batch_size, 1, 1)), dim=-1)
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'EXPAND'
    MODE_ADD = 'ADD'
    MODE_CONCAT = 'CONCAT'

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def __init__(self,
                 dim,
                 dropout=0.0,
                 max_len=512,
                 mode=MODE_ADD):
        """
        Learned encoding of postion, trainable
        :param dim: Embedding dimension (equal to the input dimension in case of mode ADD)
        :param dropout: Dropout rate of final output
        :param max_len: Maximum len of input sequence length
        :param mode: Type of merging input and position encoding
        """
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = max_len
        self.embedding_dim = dim
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.tensor(self.num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.tensor(self.num_embeddings, embedding_dim))
        self.reset_parameters()

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return self.dropout(F.embedding(indices.type(torch.LongTensor), self.weight))
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return self.dropout(x + embeddings)
        if self.mode == self.MODE_CONCAT:
            return self.dropout(torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1))
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )


if __name__ == '__main__':
    pe = PositionEncoding(50, 0, mode="ADD")
    # pe = PositionEmbedding(50, 0, mode="ADD")
    plt.figure(figsize=(15, 5))

    y = pe.forward(torch.zeros((3, 100, 50), requires_grad=True))
    print(y.shape)
    print(y)
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.savefig("psecd.jpg")
