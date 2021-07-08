# Transformer with einsum 
Motivation form https://rockt.github.io/2018/04/30/einsum

Einsum is convenient way to do matrix manipulation and should be used more widely.

Note: the current implementation of einsum is slow in pytorch and there are works to be done to make it faster.

Based on https://github.com/hyunwoongko/transformer

## Requirements:

Dataset : Multi30K - German - English traslation : https://arxiv.org/abs/1605.00459

Dataset is loaded by torchtext(so you don't have to download anythings)

1. Install torch by https://pytorch.org/get-started/locally/
2. Install requirements.txt
3. Download token data for German and English
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

##Train Model : 
```
python train.py
```

###1 Implementation difference from the original 

### Build attention mechanism base on einsum method 

```python
class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout=0):
        """
        Implementation of Scale dot product attention with einsum
        """
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, e=1e-12):

        # cover attention and multi-head attention
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

```

### Re-build position encoding.

```python
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

```
<br><br>

### Add position embedding (learned layer)

```python
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

```

