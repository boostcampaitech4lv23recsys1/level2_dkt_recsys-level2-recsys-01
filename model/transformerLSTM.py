from model import Transformer
import torch.nn as nn
import torch


class TransformerLSTM(Transformer):
    def __init__(
        self,
        dim_model,
        dim_ffn,
        num_heads,
        n_layers,
        dropout_rate,
        embedding_dim,
        device,
        config,
    ):
        super().__init__(
            dim_model,
            dim_ffn,
            num_heads,
            n_layers,
            dropout_rate,
            embedding_dim,
            device,
            config,
        )
        self.lstm = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=n_layers,  # 이거 lstm이랑 transformer랑 달라야하지않나?
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False,
        )

    def forward(self, X):
        # Embedding
        cat_feature = X["cat"].to(self.device)
        num_feature = X["num"].to(self.device)
        mask = X["mask"]

        # past
        cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(
                self.emb_cat_dict[cat_col](cat_feature[:, :, idx])
            )  # 데이터에 따라 수정

        cat_emb = torch.cat(cat_emb_list, dim=-1)
        cat_emb = self.cat_comb_proj(cat_emb)

        num_emb = self.num_comb_proj(num_feature[:, :, :-1])  # 마스크를 빼고 넣는다.
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

        out = self.encoder(X, mask)
        hidden_out, cell_out = self.lstm(out)
        out = self.prediction(hidden_out)

        return out.squeeze(2)
