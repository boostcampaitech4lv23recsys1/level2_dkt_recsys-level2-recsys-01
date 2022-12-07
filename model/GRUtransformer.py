from model import LSTM
import torch.nn as nn
import torch
from model.layer.encoder import Encoder
from model.utils import get_attn_mask, feature_embedding, feature_one_embedding


class GRUtransformer(LSTM):
    def __init__(self, config):
        super().__init__(config)
        self.modelargs = config["arch"]["args"]
        self.GRU = nn.GRU(
            input_size=self.modelargs["dim_model"],
            hidden_size=self.modelargs["dim_model"],
            num_layers=self.modelargs["n_layers_GRU"],
            batch_first=True,
            dropout=self.modelargs["dropout_rate"],
            bidirectional=False,
        )
        self.encoder = Encoder(
            dim_model=self.modelargs["dim_model"],
            dim_ffn=self.modelargs["dim_ffn"],
            num_heads=self.modelargs["num_heads"],
            dropout_rate=self.modelargs["dropout_rate"],
            n_layers=self.modelargs["n_layers_transformer"],
        )
        
    def forward(self, X):
        mask = X["mask"]
        if self.one_embedding:
            X = feature_one_embedding(
                X=X,
                cat_comb_proj=self.cat_comb_proj,
                num_comb_proj=self.num_comb_proj,
                emb_cat=self.emb_cat,
                device=self.device
            )
        else:
            X = feature_embedding(
                X=X,
                cat_cols=self.cat_cols,
                emb_cat_dict=self.emb_cat_dict,
                cat_comb_proj=self.cat_comb_proj,
                num_comb_proj=self.num_comb_proj,
                device=self.device,
            )

        mask = get_attn_mask(mask).to(self.device)

        hidden_out, _ = self.GRU(self.dropout(X))
        out = self.encoder(hidden_out, mask)
        out = self.prediction(out)
        
        return out.squeeze(2)