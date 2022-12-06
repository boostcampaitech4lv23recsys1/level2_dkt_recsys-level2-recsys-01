from model import LSTM
import torch.nn as nn
import torch
from model.layer.encoder import Encoder


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
        
    def forward(self, input):
        cat_feature = input['cat'].to(self.device)
        num_feature = input['num'].to(self.device)
        mask = input["mask"]
        
        cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(self.emb_cat_dict[cat_col](cat_feature[:, :, idx]))

        cat_emb = torch.cat(cat_emb_list, dim = -1)
        cat_emb = self.cat_comb_proj(cat_emb)
        num_emb = self.num_comb_proj(num_feature)
        X = torch.cat([cat_emb, num_emb], -1)
        
        mask_pad = (
            torch.BoolTensor(mask == 1).unsqueeze(1).unsqueeze(1)
        )  # (batch_size, 1, 1, max_len)
        mask_time = (
            1 - torch.triu(torch.ones((1, 1, mask.size(1), mask.size(1))), diagonal=1)
        ).bool()  # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to(
            self.device
        )  # (batch_size, 1, max_len, max_len)

        hidden_out, _ = self.GRU(self.dropout(X))
        out = self.encoder(hidden_out, mask)
        out = self.prediction(out)
        
        return out.squeeze(2)