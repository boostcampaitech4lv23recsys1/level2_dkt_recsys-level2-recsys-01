"""
y, y_hat의 차이를 뭐로 구할것 인지?
ex) RMSE
직접적으로 줄이고자 하는 대상
"""

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def RMSE_loss(output, target):

    raise NotImplementedError
