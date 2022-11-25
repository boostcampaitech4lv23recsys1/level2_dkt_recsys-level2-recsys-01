"""
train.py에서 이 파일을 호출
주어진 data에 대해서 학습만 시키면 됨
"""
import torch
from numpy import inf
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
# from ..logger import wandb_logger
from . import loss, metric, optimizer, scheduler
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb


class BaseTrainer(object):
    """
    model과 data_loader, 그리고 각종 config를 넣어서 학습시키는 class
    """

    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        config,
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config

        self.device = config.device

        # 학습 관련 파라미터
        self.criterion = loss.get_loss(config)
        self.metric_ftns = metric.get_metric(config)
        self.optimizer = optimizer.get_optimizer(self.model, config)
        self.lr_scheduler = scheduler.get_scheduler(self.optimizer, config)

        self.epochs = config.epoch
        self.start_epoch = 1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """

        """
        할일
        1. return에 wandb로 찍을 값 넘기기
        2. 최적의 경우 모델 parameter 저장해서 inference가 가능하도록 하기 (pytorch template 원래 파일 가면 예시가 잘 있다.)
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

        self.lr_scheduler.step()

        return loss

    def train(self):
        """
        Full training logic
        """
        # 고유 키 값 넣어주세요
        key = None

        wandb_logger.init(key, self.model)
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            wandb.log(result)
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": train_loss,
                    "train_auc_epoch": train_auc,
                    "train_acc_epoch": train_acc,
                    "valid_auc_epoch": auc,
                    "valid_acc_epoch": acc,
                }
            )


class XGBoostTrainer:
    def __init__(
            self,
            model,
            train_X,
            train_y,
            test_X,
            test_y,
            features,
            early_stopping_rounds=100,
            verbose=5,
    ):

        self.model = model

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.features = features
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose


    def train(self):
        self.model.fit(
            X=self.train_X[self.features],
            y=self.train_y,
            eval_set=[(self.test_X[self.features], self.test_y)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose
        )

        preds = self.model.predict_proba(self.test_X[self.features])[:, 1]
        acc = accuracy_score(self.test_y, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(self.test_y, preds)

        print(f"VALID AUC : {auc} ACC : {acc}\n")
