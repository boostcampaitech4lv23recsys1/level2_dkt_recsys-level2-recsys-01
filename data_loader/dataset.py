import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import random

import pandas as pd


class BaseDataset(Dataset):
    def __init__(self, data, idx, config) -> None:
        super().__init__()
        self.data = data[data["userID"].isin(idx)]
        self.user_list = self.data["userID"].unique().tolist()
        self.config = config
        self.max_seq_len = config["dataset"]["max_seq_len"]

        # def grouping_data(r, column):
        #     result = []
        #     for col in column:
        #         result.append(np.array(r[col]))
        #     return result

        self.Y = self.data.groupby("userID")["answerCode"]

        self.cur_cat_col = [f"{col}2idx" for col in config["cat_cols"]] + ["userID"]
        self.cur_num_col = config["num_cols"] + ["userID"]
        self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        self.X_num = self.data.loc[:, self.cur_num_col].copy()

        self.X_cat = self.X_cat.groupby("userID")
        self.X_num = self.X_num.groupby("userID")

        # self.data = data[.loc[idx, :].reset_index(drop=True)]
        self.data = data[data['userID'].isin(idx)]
        self.user_list = self.data['userID'].unique().tolist()
        self.group_data = self.data.groupby("userID")
        self.config = config
        self.max_seq_len = config['dataset']['max_seq_len']

        # def grouping_data(r, column):
        #     result = []
        #     for col in column[:]:
        #         result.append(np.array(r[col]))
        #     return result
        
        # self.Y = self.data.apply(
        #     grouping_data, column=["answerCode"]
        # )
        # self.data = self.data.drop(["answerCode2idx"], axis=1)

        self.cur_cat_col = [f'{col}2idx' for col in config['cat_cols']] + ['userID']
        self.cur_num_col = config['num_cols'] + ['userID']

        # self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        # self.X_num = self.data.loc[:, self.cur_num_col].copy()
        #
        # self.X_cat = self.X_cat.groupby("userID") \
        #     .apply(grouping_data, column=self.cur_cat_col) \
        #     .apply(lambda x: x[:-1])
        # self.X_num = self.X_num.groupby("userID") \
        #     .apply(grouping_data, column=self.cur_num_col) \
        #     .apply(lambda x: x[:-1])

    # 총 데이터의 개수를 리턴
    def __len__(self) -> int:
        return len(self.user_list)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        user = self.user_list[index]
        cat = self.X_cat.get_group(user).values
        num = self.X_num.get_group(user).values.astype(np.float32)
        y = self.Y.get_group(user).values

        # cat_cols = [cat[i] for i in range(cat.shape[1])]
        # num_cols = [num[i].astype(str).astype(float) for i in range(len(num))]

        seq_len = cat.shape[0]
        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 놔둔다
        if seq_len >= self.max_seq_len:
            cat = torch.tensor(cat[-self.max_seq_len :], dtype=torch.long)
            num = torch.tensor(num[-self.max_seq_len :], dtype=torch.float32)
            y = torch.tensor(y[-self.max_seq_len :], dtype=torch.float32)
            mask = torch.ones(self.max_seq_len, dtype=torch.long)
        else:
            cat = torch.cat(
                (
                    torch.zeros(
                        self.max_seq_len - seq_len,
                        len(self.cur_cat_col),
                        dtype=torch.long,
                    ),
                    torch.tensor(cat, dtype=torch.long),
                )
            )
            num = torch.cat(
                (
                    torch.zeros(
                        self.max_seq_len - seq_len,
                        len(self.cur_num_col),
                        dtype=torch.float32,
                    ),
                    torch.tensor(num, dtype=torch.float32),
                )
            )
            y = torch.cat(
                (
                    torch.zeros(self.max_seq_len - seq_len, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
            )
            mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            mask[-seq_len:] = 1

        return {"cat": cat, "num": num, "answerCode": y, "mask": mask}


def collate_fn(batch):
    X_cat, X_num, y, mask = [], [], [], []
    for user in batch:
        X_cat.append(user["cat"])
        X_num.append(user["num"])
        y.append(user["answerCode"])
        mask.append(user["mask"])
    return {
        "cat": torch.stack(X_cat),
        "num": torch.stack(X_num),
        "answerCode": torch.stack(y),
        "mask": torch.stack(mask),
    }


def get_loader(train_set, val_set, config):
    train_loader = DataLoader(
        train_set,
        num_workers=config["num_workers"],
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        val_set,
        num_workers=config["num_workers"],
        shuffle=False,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


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
