import torch
import numpy as np
from torch import nn
from torch.nn import LayerNorm


class CustomLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(CustomLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        if hidden is None:
            hidden = dim
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    po = PositionwiseFeedForward(50)
    x = torch.zeros((3, 100, 50), requires_grad=True)

    out = po(x)
    print(out.shape)
