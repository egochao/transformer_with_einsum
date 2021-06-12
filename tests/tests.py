import sys
import torch

sys.path.append("..")

from modules.encoder import CustomEncoderLayer, CustomEncoder

vocab_size = 1000
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

    x = torch.randint(0, vocab_size, (300, 100))
    encode = CustomEncoder(vocab_size, max_len, d_embed, ffn_hidden, n_head, n_layers, dropout)
    out = encode(x, mask=None)
    print(out.shape)


if __name__ == '__main__':
    test_encoder()