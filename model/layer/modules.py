import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        _2i = torch.arnage(0, d_model, step=2, dtype=torch.float)

        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))

        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_len = x.size()

        return self.pe[:seq_len, :]
