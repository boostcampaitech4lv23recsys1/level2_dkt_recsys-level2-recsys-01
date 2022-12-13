import torch
import torch.nn as nn

from model.utils import *
from model.transformer import Transformer
from model.layer.encoder import Encoder
from model.layer.modules import PositionalEncoding


class GTN(Transformer):
    def __init__(
            self,
            dim_model,
            dim_ffn,
            num_heads,
            n_layers_transformer,
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
        self.max_len = config["dataset"]["max_seq_len"]

        if self.one_embedding:
            self.offset = self.config['offset']
            self.time_emb_cat = nn.Embedding(self.offset + 1, self.embedding_dim, padding_idx=0)
        else:
            self.time_embedding_cat_col = {}
            self.time_cat_col_len = self.config['cat_col_len']
            for cat_col in self.cat_cols:
                self.time_embedding_cat_col[cat_col] = nn.Embedding(self.cat_col_len[cat_col] + 1, self.embedding_dim,
                                                               padding_idx=0)
            self.time_emb_cat_dict = nn.ModuleDict(self.embedding_cat_col)

        self.time_cat_comb_proj = nn.Linear(
            self.embedding_dim * len(self.cat_cols), self.dim_model // 2
        )
        self.time_num_comb_proj = nn.Linear(len(self.num_cols), self.dim_model // 2)

        self.time_encoder = Encoder(
            dim_model=self.dim_model,
            dim_ffn=self.dim_ffn,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            n_layers=self.n_layers_transformer,
        )

        self.positional_emb = PositionalEncoding(dim_model, self.max_len)
        self.prediction = nn.Sequential(
            nn.Linear(self.dim_model * 2, self.dim_model * 2),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(self.dim_model * 2, 1),
            nn.Sigmoid(),
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

        out = self.prediction(torch.cat([channel_out, step_out], dim=-1))

        return out.sqeeuze(2)
