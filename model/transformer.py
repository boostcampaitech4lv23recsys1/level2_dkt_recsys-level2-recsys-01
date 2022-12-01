import torch
import torch.nn as nn

from model.layer.encoder import Encoder

class Transformer(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_ffn,
        num_heads,
        n_layers,
        dropout_rate,
        embedding_dim,
        device,
        config
        ):
        super().__init__()
        self.config = config
        self.device = device
        self.model_args = self.config['arch']['args']

        self.dim_model = dim_model
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.cat_cols = self.config['cat_cols']
        self.num_cols = self.config['num_cols']
        self.embedding_dim = embedding_dim
        self.embedding_cat_col = dict()
        self.cat_col_len = self.config['cat_col_len']
        self.dropout = nn.Dropout(dropout_rate)
        
        self.embedding_answercode = nn.Embedding(3, self.embedding_dim)
        for cat_col in self.cat_cols:
            self.embedding_cat_col[cat_col] = nn.Embedding(self.cat_col_len[cat_col] + 1, self.embedding_dim)
        
        self.emb_cat_dict = nn.ModuleDict(self.embedding_cat_col)
        
        self.cat_comb_proj = nn.Linear(self.embedding_dim * len(self.cat_cols), self.dim_model // 2)
        self.num_comb_proj = nn.Linear(len(self.num_cols), self.dim_model // 2)
        
        self.encoder = Encoder(dim_model = self.dim_model,
                               dim_ffn = self.dim_ffn,
                               num_heads = self.num_heads,
                               dropout_rate = self.dropout_rate,
                               n_layers = self.n_layers)
        self.prediction = nn.Sequential(nn.Linear(self.dim_model, 1), nn.Sigmoid())
    
    def forward(self, X):
        # Embedding
        breakpoint()
        cat_feature = X['cat'].to(self.device)
        num_feature = X['num'].to(self.device)

        # past
        cat_emb_list = []
        breakpoint()
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(self.emb_cat_dict[cat_col](cat_feature[:, :, idx])) # 데이터에 따라 수정

        cat_emb = torch.cat(cat_emb_list, dim = -1)
        cat_emb = self.cat_comb_proj(cat_emb)
        
        num_emb = self.num_comb_proj(num_feature[:, :, :-1]) # 마스크를 빼고 넣는다.
        X = torch.cat([cat_emb, num_emb], -1)
        
        out, _ = self.encoder(X)
        out = self.prediction(out)
        
        return out.squeeze(2)