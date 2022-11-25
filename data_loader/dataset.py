import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

        self.Y = self.data.groupby("userID").apply(
            grouping_data, column=["no_meaning", "answerCode"]
        )
        self.data = self.data.drop(["answerCode"], axis=1)

        base_cat_col = set(
            [
                "KnowledgeTag2idx",
                "question_number2idx",
                "test_cat2idx",
                "test_id2idx",
                "test_month2idx",
                "test_day2idx",
                "test_hour2idx",
            ]
        )

        self.cur_num_col = list(set(self.data.columns) - base_cat_col)
        self.cur_cat_col = list(set(self.data.columns) - set(self.cur_num_col)) + [
            "userID"
        ]

        self.X_cat = self.data.loc[:, self.cur_cat_col].copy()
        self.X_num = self.data.loc[:, self.cur_num_col].copy()

        self.X_cat = (
            self.X_cat.groupby("userID")
            .apply(grouping_data, column=self.cur_cat_col)
            .apply(lambda x: x[:-1])
        )
        self.X_num = (
            self.X_num.groupby("userID")
            .apply(grouping_data, column=self.cur_num_col)
            .apply(lambda x: x[:-1])
        )

    # 총 데이터의 개수를 리턴
    def __len__(self) -> int:
        return len(self.data_X)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        cat = self.X_cat[index][0]
        num = self.X_num[index][0]
        y = torch.tensor(self.X_num[index][0], dtype=np.int16)

        cat_cols = [cat[i] for i in range(len(cat))]
        num_cols = [num[i] for i in range(len(num))]

        seq_len = len(cat[0])

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 놔둔다
        if seq_len > self.config.max_seq_len:
            for i, col in enumerate(cat_cols):
                cat_cols[i] = col[self.config.max_seq_len :]
            for i, col in enumerate(num_cols):
                num_cols[i] = col[-self.config.max_seq_len :]
            mask = np.ones(self.config.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.config.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cat_cols.append(mask)
        num_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cat_cols):
            cat_cols[i] = torch.tensor(col)
        for i, col in enumerate(num_cols):
            num_cols[i] = torch.tensor(col)

        return {"cat": cat_cols, "num": num_cols, "answerCode": y}


def sequence_collate_fn(batch):
    collate_X_cat = []
    collate_X_num = []
    collate_answerCode = []

    for data in batch:
        batch_X_cat = data["cat"]
        batch_X_num = data["num"]
        batch_y = data["answerCode"]

        collate_X_cat.append(batch_X_cat)
        collate_X_num.append(batch_X_num)
        collate_answerCode.append(batch_y)

    collate_X_cat = [torch.nn.utils.rnn.pad_sequence(collate_X_cat, batch_first=True)]
    collate_X_num = [torch.nn.utils.rnn.pad_sequence(collate_X_num, batch_first=True)]
    collate_answerCode = [
        torch.nn.utils.rnn.pad_sequence(collate_answerCode, batch_first=True)
    ]

    return {
        "cat": torch.stack(collate_X_cat),
        "num": torch.stack(collate_X_num),
        "y": torch.stack(collate_answerCode),
    }


def get_loader(train_set, val_set, collate=sequence_collate_fn):
    train_loader = DataLoader(
        train_set,
        num_workers=CFG.num_workers,
        shuffle=True,
        batch_size=CFG.batch_size,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        val_set,
        num_workers=CFG.num_workers,
        shuffle=False,
        batch_size=CFG.batch_size,
        collate_fn=collate,
    )
    return train_loader, valid_loader
