import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import random

import pandas as pd

class BaseDataset(Dataset):
    def __init__(
            self,
            data: Dataset,
            idx: list,
            config: dict,
    ) -> None:
        super().__init__()
        self.data = data[data["userID"].isin(idx)]
        self.user_list = self.data["userID"].unique().tolist()
        self.config = config
        self.max_seq_len = config["dataset"]["max_seq_len"]

        self.Y = self.data.groupby("userID")["answerCode"]

        self.cur_cat_col = [f"{col}2idx" for col in config["cat_cols"]] + ["userID"]
        self.cur_num_col = config["num_cols"] + ["userID"]
        self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        self.X_num = self.data.loc[:, self.cur_num_col].copy()

        self.X_cat = self.X_cat.groupby("userID")
        self.X_num = self.X_num.groupby("userID")

        self.group_data = self.data.groupby("userID")

    def __len__(self) -> int:
        """
        return data length
        """
        return len(self.user_list)

    def __getitem__(self, index: int) -> object:
        user = self.user_list[index]
        cat = self.X_cat.get_group(user).values[:, :-1]
        num = self.X_num.get_group(user).values[:, :-1].astype(np.float32)
        y = self.Y.get_group(user).values
        seq_len = cat.shape[0]

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
                        len(self.cur_cat_col) - 1,
                        dtype=torch.long,
                    ),
                    torch.tensor(cat, dtype=torch.long),
                )
            )
            num = torch.cat(
                (
                    torch.zeros(
                        self.max_seq_len - seq_len,
                        len(self.cur_num_col) - 1,
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
    """
    [batch, data_len, dict] -> [dict, batch, data_len]
    """
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


def get_loader(
        train_set: Dataset,
        val_set: Dataset,
        config: dict
) -> DataLoader:
    """
    get Data Loader
    """
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
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader


class XGBoostDataset(object):
    def __init__(
        self,
        config: dict,
    ) -> None:
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
