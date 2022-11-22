from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from ..config import CFG

class Preprocess:
    def __init__(self, CFG):
        self.config = CFG
        self.train_data = None
        self.test_data = None
    
    ## preprocessing
    def __preprocessing(data):
        raise NotImplementedError
    
    ## feature engineering
    def __feature_engineering(data):
        raise NotImplementedError
    
    ## 데이터 불러오기
    def grouping_data(r, column):
        result = []
        for col in column[1:]:
            result.append(np.array(r[col]))
        return result
        
    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.config.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = df.clolumns.tolist()
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                self.grouping_data, column=columns
                )
            )

        return group.values
    
    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class BaseDataset(Dataset):
    def __init__(self, data, config) -> None:
        super().__init__()
        self.data = data
        self.config = config
    
    # 총 데이터의 개수를 리턴
    def __len__(self) -> int:
        return len(self.data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        row = self.data[index]
        
        
        return
        # return (self.user_id[index], self.item_id[index], self.rating[index])
