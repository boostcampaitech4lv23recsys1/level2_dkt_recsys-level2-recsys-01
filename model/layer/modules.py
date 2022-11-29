import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
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


class PositionwiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ffn, drop_out=0.1):
        super(PositionalEncoding, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=True),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(d_ffn, d_model, bias=True),
        )

    def forward(self, x):
        return self.sequential(x)
