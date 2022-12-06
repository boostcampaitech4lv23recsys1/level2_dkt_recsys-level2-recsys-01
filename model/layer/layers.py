import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, 
        dim_model: int, 
        dropout_rate: float
    ) -> None:
        """
        Args:
            dim_model (int): Total dimension of the model.
            dropout_rate (float): Dropout probability on ``attn_output_weights``.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.dim_model = dim_model
        self.dropout = nn.Dropout(dropout_rate)
        
        
    def forward(
        self, 
        q: torch.tensor, 
        k: torch.tensor, 
        v: torch.tensor, 
        attn_mask=None
    ) -> torch.tensor:
        """
        Args:
            q (torch.tensor): Query embeddings of shape :math:`(b, L, E_q)`. Queries are compared against key-value pairs to produce the output.
            k (torch.tensor): Key embeddings of shape :math:`(b, S, E_k)`
            v (torch.tensor): Value embeddings of shape :math:`(b, S, E_v)`
            attn_mask (torch.tensor, optional): Masks where zero padded. If specified, a 2D or 3D mask preventing attention to certain positions. Defaults to None.

        Returns:
            output: Attention outputs of shape :math:`(L, E)` Where :math:`L` is the target sequence length, and :math:`E` is the
          embedding dimension ``embed_dim``.
            attn: Attention weight.
        """
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_model)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            dim_model: int,
            dropout_rate: float
    ) -> None:
        """
        Args:
            num_heads (int): Number of parallel attention heads. Note that ``embed_dim`` will be split.
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            dim_model (int): Total dimension of the model.
            dropout_rate (float): Dropout probability on ``attn_output_weights``.
        """
        assert dim_model % num_heads == 0
        super(MultiHeadAttention, self).__init__()

        self.dim_head = dim_model // num_heads
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.attention = ScaledDotProductAttention(
            dim_model=dim_model, dropout_rate=dropout_rate
        )

        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)
        self.w_o = nn.Linear(dim_model, dim_model)

    def forward(
        self,
        q: torch.tensor,
        k: torch.tensor,
        v: torch.tensor,
        attn_mask=None
    ) -> torch.tensor:
        """
        Args:
            q (torch.tensor): Query embeddings of shape :math:`(b, L, E_q)`. Queries are compared against key-value pairs to produce the output.
            k (torch.tensor): Key embeddings of shape :math:`(b, S, E_k)`
            v (torch.tensor): Value embeddings of shape :math:`(b, S, E_v)`
            attn_mask (torch.tensor, optional): Masks where zero padded. If specified, a 2D or 3D mask preventing attention to certain positions. Defaults to None.

        Returns:
            output: Attention outputs of shape :math:`(L, E)` Where :math:`L` is the target sequence length, and :math:`E` is the
          embedding dimension ``embed_dim``.
            attn: Attention weight.
        """
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

        output = self.w_o(output)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    def __init__(
            self,
            dim_model: int,
            dim_ffn: int,
            dropout_rate: float
    ) -> None:
        """
        Args:
            dim_model (int): Total dimension of the model.
            dim_ffn (int): dimension of feed forward net.
            dropout_rate (float): Dropout probability on ``attn_output_weights``.
        """
        super(PositionWiseFeedForward, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(dim_model, dim_ffn, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(dim_ffn, dim_model, bias=True),
        )

    def forward(self, x):
        return self.sequential(x)


class EncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int,
            dim_ffn: int,
            num_heads: int,
            dropout_rate=0.1
    ) -> None:
        """
        Args:
            dim_model (int): Total dimension of the model.
            dim_ffn (int): dimension of feed forward net.
            num_heads (int): Number of parallel attention heads. Note that ``embed_dim`` will be split.
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            dropout_rate (float): Dropout probability on ``attn_output_weights``.
        """
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(
            dim_model=dim_model, num_heads=num_heads, dropout_rate=dropout_rate
        )
        self.norm_attn = nn.LayerNorm(dim_model)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.ffn = PositionWiseFeedForward(
            dim_model=dim_model, dim_ffn=dim_ffn, dropout_rate=dropout_rate
        )
        self.norm_ffn = nn.LayerNorm(dim_model)
        self.dropout_ffn = nn.Dropout(p=dropout_rate)

    def forward(
            self,
            x: torch.tensor,
            attn_mask: torch.tensor,
    ) -> torch.tensor:
        output, attn = self.attention(q=x, k=x, v=x, attn_mask=attn_mask)
        output = self.norm_attn(self.dropout_attn(output) + x)

        x = output
        output = self.ffn(output)
        output = self.norm_ffn(self.dropout_ffn(output) + x)

        return output
