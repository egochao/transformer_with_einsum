import torch
from torch import nn
import numpy as np


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout=0):
        """
        Implementation of Scale dot product attention with einsum
        """
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, e=1e-12):

        if len(q.size()) == 4:
            batch, head, length, dim = k.size()
            product = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        else:
            batch, length, dim = k.size()
            product = torch.einsum('b i d, b j d -> b i j', q, k)
        scale_product = product * dim ** -0.5

        if mask is not None:
            scale_product = scale_product.masked_fill(mask == 0, e)

        attention = self.softmax(scale_product)
        if len(q.size()) == 4:
            output = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        else:
            output = torch.einsum('b i j, b j d -> b i d', attention, v)

        if self.dropout != 0:
            output = self.dropout(output)
        return output, attention


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, bias=True):
        """
        Multi Head attention class
        :param dim: Attention dimension of input (last dimension)
        :param n_head: Number of head
        :param bias: Use bias or not
        """
        super(CustomMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.dim = dim
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(dim, dim, bias=bias)
        self.w_k = nn.Linear(dim, dim, bias=bias)
        self.w_v = nn.Linear(dim, dim, bias=bias)

        self.w_cat = nn.Linear(dim, dim, bias=bias)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask.unsqueeze(1)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # split dim to n_head
        q, k, v = self.split(q), self.split(k), self.split(v)
        # print(v)
        out, attention = self.attention(q, k, v, mask=mask)
        # print(out)
        out = self.concat(out)
        out = self.w_cat(out)
        return out

    def split(self, tensor):
        """
        Split tensor into n_head by breaking the embedding into n_head vectors
        :param tensor: Input 3 dimension tensor
        :return: Output 4 dimension tensor with n_head
        """
        batch, length, d_input = tensor.size()
        assert d_input == self.dim, "Input dimension mismatch with defined MultiHeadAttention"
        d_tensor = d_input // self.n_head
        # print(d_tensor, d_input, self.n_head)
        tensor = tensor.view(batch, self.n_head, length, d_tensor)
        return tensor

    def concat(self, tensor):
        """
        Concat all head to the original size tensor
        Inverse function of split
        :param tensor: Multi-head tensor 4d
        :return: Concatenated tensor 3d
        """
        batch, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.view(batch, length, d_model)
        return tensor


if __name__ == '__main__':
    sc = ScaleDotProductAttention()
    mh = CustomMultiHeadAttention(50, 5)
    # x = torch.zeros((3, 3, 100, 50), requires_grad=True)
    x = torch.zeros((3, 100, 50), requires_grad=True)

    out = sc(x, x, x)
    print(out[0].shape, "scale_product")

    out = mh(x, x, x)
    print(out.shape, "multi head attention")
