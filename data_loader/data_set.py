from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        # return len(self.user_id)
        return

    def __getitem__(self, index: int) -> object:
        return
        # return (self.user_id[index], self.item_id[index], self.rating[index])
