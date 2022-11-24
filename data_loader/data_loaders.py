"""
dataset과 config를 넣어주면
batch, valid size, 등등 알아서 해주는 data loader
"""

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset: Dataset, config, collate_fn=default_collate):
        """
        dataset이랑 config 줘서 dataloader 만들기
        """
        self.n_samples = len(dataset)
        self.batch_size = config.batch_size
        self.valid

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "collate_fn": collate_fn,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
