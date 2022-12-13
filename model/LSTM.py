import torch
import torch.nn as nn

from model.utils import feature_embedding, feature_one_embedding

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.device = config['device']
        self.model_args = self.config['arch']['args']

        self.hidden_dim = self.model_args['hidden_dim']
        self.n_layers = self.model_args['n_layers']
        self.dropout_rate = self.model_args['dropout_rate']

        self.cat_cols = self.config['cat_cols']
        self.num_cols = self.config['num_cols']

        self.embedding_dim = self.model_args['embedding_dim']

        self.one_embedding = self.config['one_embedding']
        if self.one_embedding:
            self.offset = self.config['offset']
            self.emb_cat = nn.Embedding(self.offset+1, self.embedding_dim, padding_idx=0)
        else:
            self.embedding_cat_col = {}
            self.cat_col_len = self.config['cat_col_len']
            for cat_col in self.cat_cols:
                self.embedding_cat_col[cat_col] = nn.Embedding(self.cat_col_len[cat_col] + 1, self.embedding_dim, padding_idx=0)
            self.emb_cat_dict = nn.ModuleDict(self.embedding_cat_col)
        
        self.embedding_answercode = nn.Embedding(3, self.embedding_dim, padding_idx=0)
        
        self.cat_comb_proj = nn.Sequential(
            nn.Linear(self.embedding_dim*len(self.cat_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )
        self.num_comb_proj = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_rate
        )

        self.prediction = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, X):
        
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

        out, _ = self.lstm(self.dropout(X))
        out = self.prediction(out)
        
        return out.squeeze(2)
    