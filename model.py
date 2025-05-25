import torch
from torch import nn
import matplotlib.pyplot as plt
"""
    input shape [batch, seq_len, d_model]
"""

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        # position encoding shape 应该和输入的shape一致， shape [max_seq_len, 1]
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        item  = 1/10000 ** (torch.arange(0, d_model, 2))

        tmp_pos = position * item
        pe = torch.zeros(max_seq_len, d_model)

        pe[:,0::2] = torch.sin(tmp_pos)
        pe[:,1::2] = torch.cos(tmp_pos)

        plt.matshow(pe)
        plt.show()

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe, False)
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        pe = self.pe
        return x + pe[:,:seq_len,:]

if __name__ == '__main__':
    PositionEncoding(512, 100)