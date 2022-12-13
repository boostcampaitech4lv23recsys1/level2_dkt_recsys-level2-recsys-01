from .layers import EncoderLayer
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_ffn: int,
        num_heads: int,
        dropout_rate: float,
        n_layers: int,
    ) -> None:
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim_model=dim_model,
                    dim_ffn=dim_ffn,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, attn_mask):
        for encoder_layer in self.layers:
            x = encoder_layer(x, attn_mask)

        return x
