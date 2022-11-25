from torch.utils.data import Dataset
from sklearn import preprocessing
import random

import pandas as pd

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
        
        # hi
        return
        # return (self.user_id[index], self.item_id[index], self.rating[index])


class XGBoostDataset(object):
    def __init__(
            self,
            config,
    ):
        self.config = config
        self.train = self._load_train_data()
        self.test = self._load_test_data()
        self.user_ids = list(self.train["userID"].unique())
        self.test_ids = list(self.test["userID"].unique())
        self.__preprocessing()

    def __preprocessing(self):
        cat_cols = self.config.cat_cols

        label_encoder = preprocessing.LabelEncoder()
        for cat_col in cat_cols:
            self.train[cat_col] = label_encoder.fit_transform(self.train[cat_col])

        self.train = self.train.reset_index(drop=True)

    def _split(
            self,
            ratio=0.9,
    ):
        random.seed(42)

        train_df = self.train[~self.train["answerCode"] == -1]

        random.shuffle(self.user_ids)

        train_len = int(len(self.user_ids) * ratio)
        train_ids = self.user_ids[:train_len]
        test_ids = self.user_ids[train_len:]

        train = train_df[train_df["userID"].isin(train_ids)]
        valid = train_df[train_df["userID"].isin(test_ids)]

        valid = valid[valid["userID"] != valid["userID"].shift(-1)]

        return train, valid

    def _load_train_data(self):
        return pd.read_csv(self.config.data_path)

    def _load_test_data(self):
        return pd.read_csv(self.config.test_path)


