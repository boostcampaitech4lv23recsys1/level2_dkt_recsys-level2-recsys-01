from model import TransformerLSTM
import torch.nn as nn


class TransformerGRU(TransformerLSTM):
    def __init__(
        self,
        dim_model,
        dim_ffn,
        num_heads,
        n_layers_transformer,
        n_layers_LSTM,
        dropout_rate,
        embedding_dim,
        device,
        config,
    ):
        super().__init__(
            dim_model,
            dim_ffn,
            num_heads,
            n_layers_transformer,
            n_layers_LSTM,
            dropout_rate,
            embedding_dim,
            device,
            config,
        )
        self.lstm = nn.GRU(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=n_layers_LSTM,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )
