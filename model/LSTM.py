import torch
import torch.nn as nn


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

        # Embedding
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

    def forward(self, input):

        # Embedding
        cat_feature = input['cat'].to(self.device)
        num_feature = input['num'].to(self.device)
        
        if self.one_embedding:
            batch_size, max_seq_len, _ = cat_feature.size()
            cat_emb = self.emb_cat(cat_feature).view(batch_size, max_seq_len, -1)
        else:
            cat_emb_list = []
            for idx, cat_col in enumerate(self.cat_cols):
                cat_emb_list.append(self.emb_cat_dict[cat_col](cat_feature[:, :, idx]))
            cat_emb = torch.cat(cat_emb_list, dim = -1)

        cat_emb = self.cat_comb_proj(cat_emb)
        num_emb = self.num_comb_proj(num_feature) 

        X = torch.cat([cat_emb, num_emb], -1)

        out, _ = self.lstm(self.dropout(X))
        out = self.prediction(out)
        
        return out.squeeze(2)