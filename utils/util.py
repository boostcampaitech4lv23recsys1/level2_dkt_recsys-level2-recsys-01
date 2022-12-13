import os
import random
import json

from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch


def set_seed(seed=417):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


FEATURES = [
    "userID",
    "assessmentItemID",
    "testId",
    "week_num",
    "elapsed_time",
    "time_question_median",
    "time_user_median",
    "time_user_mean",
    "KnowledgeTag",
    "test_cat",
    "test_id",
    "question_number",
    "question_numslen",
    "test_month",
    "test_day",
    "test_hour",
    "user_acc",
    "test_acc",
    "tag_acc",
    "question_acc",
    "month_acc",
    "hour_acc",
    "exp_test",
    "exp_tag",
    "exp_question",
    "ans_cumsum",
    "continuous_user",
    "continuous_test",
]
