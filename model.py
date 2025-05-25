import torch
from sympy import transpose
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
        item  = 1/10000 ** (torch.arange(0, d_model, 2)/d_model)

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

def attention(query, key, value, mask=None):
    d_model = key.shape[-1]
    # 这里是没有用多头 query, key, value shape [batch, seq_len, d_model]
    att_ = torch.matmul(query,key.transpose(-2,-1))
    att_ = att_/d_model ** 0.5
    # 有mask的时候，设置masked_fill设置一个很小的值，这样softmax处理后就基本上趋近为0，注意力就不会注意
    if mask is not None:
        att_ = att_.masked_fill(mask, -1e9)

    att_score = torch.softmax(att_, -1)
    return torch.matmul(att_score, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()
        assert d_model % heads == 0
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False) #多头注意力后整合输出到 Add &  Norm
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads

    def forward(self,q,k,v,mask=None):
        # [batch, seq_len, d_model] -> [batch, head, seq_len, d_model]
        q = self.q_linear(q).reshape(q.shape[0], -1, self.heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).reshape(k.shape[0], -1, self.heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).reshape(v.shape[0], -1, self.heads, self.d_k).transpose(1,2)
        out = attention(q,k,v,mask)
        out = out.transpose(1,2).reshape(out.shape[0], -1, self.d_model)
        out = self.linear(out)
        out = self.dropout(out)
        return out


if __name__ == '__main__':
    PositionEncoding(512, 100)
    att = MultiHeadAttention(8,512,0.2)
    x = torch.randn(4,100,512)
    out = att(x,x,x)
    print(out.shape)