import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args['hidden_dim']
        self.n_layers = self.args['n_layers']
        self.dropout_rate = self.args['dropout_rate']
        self.cat_cols = args['cat_cols']
        self.num_cols = args['num_cols']
        self.embedding_dim = self.args['embedding_dim']
        self.embedding_cat_col = dict()
        self.cat_col_len = self.args['cat_col_len']
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.embedding_answercode = nn.Embedding(3, self.embedding_dim)
        for cat_col in self.cat_cols:
            self.embedding_cat_col[cat_col] = nn.Embedding(self.cat_col_len[cat_col] + 1, self.embedding_dim)
        
        self.emb_cat_dict = nn.ModuleDict(self.embedding_cat_col)
        
        self.cat_comb_proj = nn.Linear(self.embedding_dim * (len(self.cat_cols) + 1), self.hidden_dim // 2)
        self.num_comb_proj = nn.Linear(len(self.num_cols), self.hidden_dim // 2)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout = self.dropout_rate
        )

        # Fully connected layer
        self.prediction = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        

    def forward(self, input):

        # test, question, tag, _, mask, interaction = input

        # batch_size = input['y'].size(0)
 
        # Embedding
        cat_feature = input['cat'].to(self.device)
        num_feature = input['num'].to(self.device)

        # past
        cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(self.emb_cat_dict[cat_col](cat_feature[:, :, idx])) # 데이터에 따라 수정

        cat_emb = torch.concat(cat_emb_list, dim = -1)
        cat_emb = self.cat_comb_proj(cat_emb)
        
        num_emb = self.num_comb_proj(num_feature[:, :, :-1]) # 마스크를 빼고 넣는다.
        X = torch.cat([cat_emb, num_emb], -1)

        out, _ = self.lstm(self.dropout(X))
        out = self.prediction(out)
        
        return out