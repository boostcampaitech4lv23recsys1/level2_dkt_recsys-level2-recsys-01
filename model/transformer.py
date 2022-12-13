import torch
import torch.nn as nn

from model.utils import *
from model.layer.encoder import Encoder


class Transformer(nn.Module):
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
        super().__init__()
        self.config = config
        self.device = device
        self.model_args = self.config["arch"]["args"]

        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.n_layers_transformer = n_layers_transformer
        self.dropout_rate = dropout_rate
        self.cat_cols = self.config["cat_cols"]
        self.num_cols = self.config["num_cols"]
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout_rate)

        self.one_embedding = self.config['one_embedding']
        if self.one_embedding:
            self.offset = self.config['offset']
            self.emb_cat = nn.Embedding(self.offset + 1, self.embedding_dim, padding_idx=0)
        else:
            self.embedding_cat_col = {}
            self.cat_col_len = self.config['cat_col_len']
            for cat_col in self.cat_cols:
                self.embedding_cat_col[cat_col] = nn.Embedding(self.cat_col_len[cat_col] + 1, self.embedding_dim,
                                                               padding_idx=0)
            self.emb_cat_dict = nn.ModuleDict(self.embedding_cat_col)

        self.cat_comb_proj = nn.Linear(
            self.embedding_dim * len(self.cat_cols), self.dim_model // 2
        )
        self.num_comb_proj = nn.Linear(len(self.num_cols), self.dim_model // 2)

        self.encoder = Encoder(
            dim_model=self.dim_model,
            dim_ffn=self.dim_ffn,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            n_layers=self.n_layers_transformer,
        )
        self.prediction = nn.Sequential(nn.Linear(self.dim_model, 1), nn.Sigmoid())
    
    def forward(self, X):
        mask = X['mask']

        if self.one_embedding:
            X = feature_one_embedding(
                X,
                self.cat_comb_proj,
                self.num_comb_proj,
                self.emb_cat,
                self.device)
        else:
            X = feature_embedding(
                X,
                self.cat_cols,
                self.emb_cat_dict,
                self.cat_comb_proj,
                self.num_comb_proj,
                self.device)

        mask = get_attn_mask(mask).to(self.device)

        out = self.encoder(X, mask)
        out = self.prediction(out)

        return out.squeeze(2)
