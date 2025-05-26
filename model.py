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
    def __init__(self, heads, d_model, dropout=0.1):
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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x = self.ffn(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_multi_head_attention = MultiHeadAttention(heads,d_model,dropout)
        self.fnn = FeedForward(d_model, d_ff, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        multi_head_attention_out = self.self_multi_head_attention(x, x, x, mask)
        add_norm_out_1 = self.norms[0](x + multi_head_attention_out)
        feed_forward_out = self.fnn(add_norm_out_1)
        add_norm_out_2 = self.norms[1](add_norm_out_1 + feed_forward_out)
        out = self.dropout(add_norm_out_2)
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, num_layers,heads, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encode = PositionEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(heads, d_model, d_ff, dropout) for _ in range(num_layers)])

    # 在自然语言处理任务中，为了支持 batch 计算，不同长度的句子通常会被填充（padding）到相同的长度。填充的部分（如用 pad_idx 表示）是没有实际语义信息的，因此在进行注意力（self-attention）计算时应该忽略这些位置
    def forward(self, x, src_mask=None):
        embed_x = self.embedding(x)
        pos_encode_x = self.position_encode(embed_x)
        for layer in self.encoder_layers:
            pos_encode_x = layer(pos_encode_x, src_mask)
        return pos_encode_x

class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.masked_att = MultiHeadAttention(heads, d_model, dropout)
        self.encoder_decoder_att = MultiHeadAttention(heads, d_model,dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(3)])
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    #dst_mask:decoder mask遮盖未来信息，src_dst_mask encoder-decoder mask 遮盖padding无用信息
    def forward(self, x, encode_kv, dst_mask=None, src_dst_mask=None):
        mask_att_out = self.masked_att(x,x,x,dst_mask)
        add_norm_out_1 = self.norms[0](x+mask_att_out)
        encoder_decoder_att_out = self.encoder_decoder_att(add_norm_out_1,encode_kv,encode_kv,src_dst_mask)
        add_norm_out_2 = self.norms[1](add_norm_out_1+encoder_decoder_att_out)
        ffn_out = self.ffn(add_norm_out_2)
        add_norm_out_3 = self.norms[2](add_norm_out_2+ffn_out)
        out = self.dropout(add_norm_out_3)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, num_layers,heads, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encode = PositionEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(heads, d_model, d_ff, dropout) for _ in range(num_layers)])

    # 在自然语言处理任务中，为了支持 batch 计算，不同长度的句子通常会被填充（padding）到相同的长度。填充的部分（如用 pad_idx 表示）是没有实际语义信息的，因此在进行注意力（self-attention）计算时应该忽略这些位置
    def forward(self, x, encode_kv, dst_mask=None, src_dst_mask=None):
        embed_x = self.embedding(x)
        pos_encode_x = self.position_encode(embed_x)
        for layer in self.decoder_layers:
            pos_encode_x = layer(pos_encode_x, encode_kv, dst_mask, src_dst_mask)
        return pos_encode_x


class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout=0.1,
                 max_seq_len=512):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(dec_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout, max_seq_len)
        self.linear = nn.Linear(d_model, dec_vocab_size)
        self.pad_idx = pad_idx

    def generate_mask(self, query, key, is_triu_mask=False):
        '''
            batch,seq_len   掩码mask就是二维的，经过embedding后，q,k,v -> [batch, seq_len, d_model]
        '''
        device = query.device
        batch, seq_q = query.shape
        _, seq_k = key.shape
        # batch,head,seq_q,seq_k
        mask = (key == self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch, 1, seq_q, seq_k).to(device)
        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
            dst_triu_mask = dst_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch, 1, seq_q, seq_k).to(device)
            return mask | dst_triu_mask
        return mask

    def forward(self, src, dst):
        src_mask = self.generate_mask(src, src)
        encoder_out = self.encoder(src, src_mask)
        dst_mask = self.generate_mask(dst, dst, True)
        src_dst_mask = self.generate_mask(dst, src)
        decoder_out = self.decoder(dst, encoder_out, dst_mask, src_dst_mask)
        out = self.linear(decoder_out)
        return out


if __name__ == '__main__':
    PositionEncoding(512, 100)
    att = Transformer(100, 200, 0, 512, 6, 8, 1024, 0.1)
    x = torch.randint(0, 100, (4, 64))
    y = torch.randint(0, 200, (4, 64))
    out = att(x, y)
    print(out.shape)