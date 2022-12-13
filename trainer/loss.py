"""
y, y_hat의 차이를 뭐로 구할것 인지?
ex) RMSE
직접적으로 줄이고자 하는 대상
"""

import torch.nn.functional as F
import torch.nn as nn


def BCE_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)


def nll_loss(output, target):
    return F.nll_loss(output, target)


def RMSE_loss(output, target):

    raise NotImplementedError


def get_loss(config):
    if config['loss'] == "nll_loss":
        return nll_loss
    if config['loss'] == "rmse":
        return RMSE_loss
    if config['loss'] == "bce":
        return BCE_loss
