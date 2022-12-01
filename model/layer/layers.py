import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_model, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()

        self.dim_model = dim_model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_model)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout_rate):
        assert dim_model % num_heads == 0
        super(MultiHeadAttention, self).__init__()

        self.dim_head = dim_model // num_heads
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.attention = ScaledDotProductAttention(
            dim_model=dim_model, dropout_rate=dropout_rate
        )

        self.w_q = nn.Linear(dim_model, dim_model, bias=True)
        self.w_k = nn.Linear(dim_model, dim_model, bias=True)
        self.w_v = nn.Linear(dim_model, dim_model, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = v.size(0)

        q = (
            self.w_q(q)
            .view(batch_size, -1, self.num_heads, self.dim_head)
            .transpose(1, 2)
        )
        k = (
            self.w_k(k)
            .view(batch_size, -1, self.num_heads, self.dim_head)
            .transpose(1, 2)
        )
        v = (
            self.w_v(v)
            .view(batch_size, -1, self.num_heads, self.dim_head)
            .transpose(1, 2)
        )

        output, attn = self.attention(q, k, v, attn_mask)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.dim_head)
        )

        return output, attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ffn, dropout_rate):
        super(PositionWiseFeedForward, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(dim_model, dim_ffn, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_ffn, dim_model, bias=True),
        )
        # self.w1 = nn.Linear(dim_model, dim_model)
        # self.w2 = nn.Linear(dim_model, dim_model)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.sequential(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ffn, num_heads=8, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(
            dim_model=dim_model, num_heads=num_heads, dropout_rate=dropout_rate
        )

        self.ffn = PositionWiseFeedForward(
            dim_model=dim_model, dim_ffn=dim_ffn, dropout_rate=dropout_rate
        )
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, attn_mask):
        output, attn = self.attention(q=x, k=x, v=x, attn_mask=attn_mask)
        output = self.norm(output + x)
        output = self.dropout(output)

        # ffn을 단어마다 하는 경우가 있는것을 발견 학습안될때 참고
        output = self.ffn(output)
        output = self.norm(output + x)
        output = self.dropout(output)

        return output, attn
