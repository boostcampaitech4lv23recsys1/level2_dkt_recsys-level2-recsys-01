import torch
from torch.utils.data import Dataset
from config import CFG
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, data, idx) -> None:
        super().__init__()
        self.data = data.loc[idx, :].reset_index(drop=True)
        self.config = CFG
     
        def grouping_data(r, column):
            result = []
            for col in column[1:]:
                result.append(np.array(r[col]))
            return result
    
        self.Y = self.data.groupby('userID').apply(grouping_data, column=['no_meaning', 'answerCode'])
        self.data = self.data.drop(['answerCode'], axis=1)
        
        base_cat_col = set(["KnowledgeTag2idx","question_number2idx","test_cat2idx","test_id2idx","test_month2idx","test_day2idx","test_hour2idx"])
        
        self.cur_num_col = list(set(self.data.columns) - base_cat_col)
        self.cur_cat_col = list(set(self.data.columns) - set(self.cur_num_col)) + ['userID']

        self.X_cat = self.data.loc[:,self.cur_cat_col].copy()
        self.X_num = self.data.loc[:,self.cur_num_col].copy()
        
        self.X_cat = self.X_cat.groupby('userID').apply(grouping_data, column=self.cur_cat_col).apply(lambda x : x[:-1])
        self.X_num = self.X_num.groupby('userID').apply(grouping_data, column=self.cur_num_col).apply(lambda x : x[:-1])
    
    # 총 데이터의 개수를 리턴
    def __len__(self) -> int:
        return len(self.data_X)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        return self.X_cat[index][0], self.X_num[index][0], self.Y[index][0]