import sys
import torch

sys.path.append("..")

from modules.encoder import CustomEncoderLayer


def test_encoder():
    enlayer = CustomEncoderLayer(50, 5)
    x = torch.zeros((3, 100, 50), requires_grad=True)
    out = enlayer(x)
    print(out.shape)


if __name__ == '__main__':
    test_encoder()