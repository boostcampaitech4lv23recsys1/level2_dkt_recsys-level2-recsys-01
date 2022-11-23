"""
모델 자체의 평가
학습을 통해 내가 얼마나 잘했는지, 목표를 얼마나 잘 달성했는지
ex) Accuracy, NDCG, recall 등
"""

import torch
from sklearn.metrics import roc_auc_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def auc_roc(output, target):

    raise NotImplementedError


def get_metric(config):
    if config.metric == "accuracy":
        return accuracy
    if config.metric == "aucroc":
        return auc_roc
