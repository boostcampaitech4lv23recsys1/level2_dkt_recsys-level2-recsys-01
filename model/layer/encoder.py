from layers import EncoderLayer
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            dim_model,
            dim_ffn,
            num_heads,
            dropout_rate,
            n_layers,
    ):
        super(Encoder, self).__init__()

        self.embedding

        self.layers = nn.ModuleList(
            [EncoderLayer(
                dim_model=dim_model,
                dim_ffn=dim_ffn,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            ) for _ in range(n_layers)]
        )

    def forward(self, x, attn_mask):
        x = self.embed

