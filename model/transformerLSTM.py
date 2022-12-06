from model import Transformer
import torch.nn as nn
import torch
from model.utils import get_attn_mask, feature_embedding


class TransformerLSTM(Transformer):
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
            dropout_rate,
            embedding_dim,
            device,
            config,
        )
        self.lstm = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=n_layers_LSTM,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )

    def forward(self, X):
        mask = X["mask"]
        X = feature_embedding(
            X=X,
            cat_cols=self.cat_cols,
            emb_cat_dict=self.emb_cat_dict,
            cat_comb_proj=self.cat_comb_proj,
            num_comb_proj=self.num_comb_proj,
            device=self.device,
        )

        mask = get_attn_mask(mask).to(self.device)

        out = self.encoder(X, mask)
        hidden_out, cell_out = self.lstm(out)
        out = self.prediction(hidden_out)

        return out.squeeze(2)
