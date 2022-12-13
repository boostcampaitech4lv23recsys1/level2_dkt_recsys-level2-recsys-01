import torch
import torch.nn as nn

from model.GTN import GTN
from model.utils import get_attn_mask, feature_embedding, feature_one_embedding


class GtnGRU(GTN):
    def __init__(
        self,
        dim_model,
        dim_ffn,
        num_heads,
        n_layers_transformer,
        n_layers_GRU,
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
        self.channel_gru = nn.GRU(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=n_layers_GRU,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )

        self.step_gru = nn.GRU(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=n_layers_GRU,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )

    def forward(self, X):
        mask = X['mask']
        mask = get_attn_mask(mask).to(self.device)

        if self.one_embedding:
            base_X = feature_one_embedding(
                X,
                self.cat_comb_proj,
                self.num_comb_proj,
                self.emb_cat,
                self.device)
            time_X = feature_one_embedding(
                X,
                self.time_cat_comb_proj,
                self.time_num_comb_proj,
                self.time_emb_cat,
                self.device)
        else:
            base_X = feature_embedding(
                X,
                self.cat_cols,
                self.emb_cat_dict,
                self.cat_comb_proj,
                self.num_comb_proj,
                self.device)
            time_X = feature_embedding(
                X,
                self.cat_cols,
                self.time_emb_cat_dict,
                self.time_cat_comb_proj,
                self.time_num_comb_proj,
                self.device)

        time_X = time_X + self.positional_emb(X["answerCode"])

        channel_out = self.encoder(self.dropout(base_X), mask)
        step_out = self.time_encoder(self.dropout(time_X), mask)

        channel_out, _ = self.channel_gru(channel_out)
        step_out, _ = self.step_gru(step_out)

        out = self.prediction(torch.cat([self.dropout(channel_out), self.dropout(step_out)], dim=-1))

        return out.squeeze(2)
