import sys
import torch

sys.path.append("..")

from modules.encoder import CustomEncoderLayer, CustomEncoder
from modules.decoder import CustomDecoderLayer, CustomDecoder

vocab_size = 1000
vocab_trg = 1500
max_len = 512
d_embed = 32
ffn_hidden = 32
n_head = 4
n_layers = 4
dropout = 0.1


def test_encoder():
    enlayer = CustomEncoderLayer(50, 5)
    x = torch.zeros((3, 100, 50), requires_grad=True)
    out = enlayer(x)
    print(out.shape)

    x = torch.randint(0, vocab_size, (3, 100))
    encode = CustomEncoder(vocab_size, max_len, d_embed, ffn_hidden, n_head, n_layers, dropout)
    out = encode(x, mask=None)
    print(out.shape)


def test_decoder():
    delayer = CustomDecoderLayer(32, n_head=4, ffn_hidden=32, dropout=0.1)

    x = torch.zeros((3, 100, 32), requires_grad=True)
    y = torch.zeros((3, 50, 32), requires_grad=True)
    out = delayer(y, x, None, None)
    print(out.shape)

    decoder = CustomDecoder(vocab_trg, max_len, d_embed, ffn_hidden, n_head, n_layers, dropout)

    trg_label = torch.randint(0, vocab_trg, (3, 54))
    out = decoder(trg_label, x, None, None)
    print(out.shape)


if __name__ == '__main__':
    test_encoder()
    test_decoder()
