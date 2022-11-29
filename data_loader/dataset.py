import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, data, idx, config) -> None:
        super().__init__()
        self.data = data.loc[idx, :].reset_index(drop=True)
        self.config = config
        self.max_seq_len = config['dataset']['max_seq_len']

        def grouping_data(r, column):
            result = []
            for col in column[:]:
                result.append(np.array(r[col]))
            return result
        
        self.Y = self.data.groupby("userID").apply(
            grouping_data, column=["answerCode2idx"]
        )
        # self.data = self.data.drop(["answerCode2idx"], axis=1)

        self.cur_cat_col = [f'{col}2idx' for col in config['cat_cols']] + ['userID']
        self.cur_num_col = config['num_cols'] + ['userID']

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
        return len(self.X_cat)+len(self.X_num)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        cat = self.X_cat[index]
        num = self.X_num[index]

        cat_cols = [cat[i] for i in range(len(cat))]
        num_cols = [num[i].astype(str).astype(float) for i in range(len(num))]

        seq_len = len(cat[0])

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 놔둔다
        if seq_len > self.max_seq_len:
            for i, col in enumerate(cat_cols):
                cat_cols[i] = col[-self.max_seq_len :]
            for i, col in enumerate(num_cols):
                num_cols[i] = col[-self.max_seq_len :]
            y = torch.tensor(self.Y[index][-self.max_seq_len :])
            mask = np.ones(self.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1
                
        # mask도 columns 목록에 포함시킴
        cat_cols.append(mask)
        num_cols.append(mask)

        # np.array -> torch.tensor 형변환
        if seq_len > self.max_seq_len:
            for i, col in enumerate(cat_cols):
                cat_cols[i] = torch.tensor(col)
            for i, col in enumerate(num_cols):
                num_cols[i] = torch.tensor(col)
        else:
            for i, col in enumerate(cat_cols):
                cat_cols[i] = torch.cat([torch.zeros(self.max_seq_len-seq_len),
                                            torch.tensor(col)])
            for i, col in enumerate(num_cols):
                num_cols[i] = torch.cat([torch.zeros(self.max_seq_len-seq_len),
                                         torch.tensor(col)])

        return {"cat": cat_cols, "num": num_cols, "answerCode": y}


# def sequence_collate_fn(batch):
#     collate_X_cat = []
#     collate_X_num = []
#     collate_answerCode = []

#     for data in batch:
#         batch_X_cat = data["cat"]
#         batch_X_num = data["num"]
#         batch_y = data["answerCode"]

#         collate_X_cat.append(batch_X_cat)
#         collate_X_num.append(batch_X_num)
#         collate_answerCode.append(batch_y)

#     collate_X_cat = [torch.nn.utils.rnn.pad_sequence(collate_X_cat, batch_first=True)]
#     collate_X_num = [torch.nn.utils.rnn.pad_sequence(collate_X_num, batch_first=True)]
#     collate_answerCode = [
#         torch.nn.utils.rnn.pad_sequence(collate_answerCode, batch_first=True)
#     ]

#     return {
#         "cat": torch.stack(collate_X_cat),
#         "num": torch.stack(collate_X_num),
#         "y": torch.stack(collate_answerCode),
#     }


def get_loader(train_set, val_set, config):
    train_loader = DataLoader(
        train_set,
        num_workers=config['num_workers'],
        shuffle=True,
        batch_size=config['batch_size']
        # collate_fn=config['collate_fn'],
    )
    valid_loader = DataLoader(
        val_set,
        num_workers=config['num_workers'],
        shuffle=False,
        batch_size=config['batch_size']
        # collate_fn=config['collate_fn'],
    )
    return train_loader, valid_loader
