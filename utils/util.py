import os
import random

import numpy as np
import torch


def set_seed(seed=417):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


FEATURES = [
    'userID',
    'assessmentItemID',
    'testId',
    'week_num',
    'elapsed_time',
    'time_question_median',
    'time_user_median',
    'time_user_mean',
    'KnowledgeTag',
    'test_cat',
    'test_id',
    'question_number',
    'question_numslen',
    'test_month',
    'test_day',
    'test_hour',
    'user_acc',
    'test_acc',
    'tag_acc',
    'question_acc',
    'month_acc',
    'hour_acc',
    'exp_test',
    'exp_tag',
    'exp_question',
    'ans_cumsum',
    'continuous_user',
    'continuous_test'
]