"""
train.py에서 이 파일을 호출
주어진 data에 대해서 학습만 시키면 됨
"""
import os
import torch
from numpy import inf
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from . import loss, metric, optimizer, scheduler
from logger import wandb_logger
from utils import MetricTracker
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
        config: dict,
        fold: int
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config
        self.cfg_trainer = config['trainer']
        self.fold = fold

        self.device = config['device']

        # 학습 관련 파라미터
        self.criterion = loss.get_loss(config)
        self.metric_ftns = self.cfg_trainer['metric']
        self.optimizer = optimizer.get_optimizer(self.model, config['optimizer'])
        self.lr_scheduler = scheduler.get_scheduler(self.optimizer, config)

        self.train_metrics = MetricTracker('loss', *self.metric_ftns)
        self.valid_metrics = MetricTracker('loss', *self.metric_ftns)

        self.epochs = self.cfg_trainer['epochs']
        self.start_epoch = 1
        
        self.save_dir = self.cfg_trainer['save_dir']
        self.best_val_auc = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        log = dict()
        self.model.train()
        for data in self.train_data_loader:
            target = data['answerCode'].to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                ftns = metric.get_metric(met)
                self.train_metrics.update(met, ftns(output, target))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_log = self.train_metrics.result()
        log.update(**{'train_'+k : v for k, v in train_log.items()})

        self.model.eval()
        for data in self.valid_data_loader:
            target = data['answerCode'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                ftns = metric.get_metric(met)
                self.valid_metrics.update(met, ftns(output, target))

        val_log = self.valid_metrics.result()
        log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def train(self):
        """
        Full training logic
        """
        # 고유 키 값 넣어주세요
        key = '412d7505a821bf8637059949cb5119361aa83e80'
        
        wandb_logger.init(key, self.model, self.config)
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            wandb.log(result, step=epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if result['val_aucroc'] > self.best_val_auc:
                self.best_val_auc = result['val_aucroc']
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        model_name = type(self.model).__name__
        state = {
            'model_name': model_name,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
        }
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'fold_{self.fold}_best_model.pt')
        torch.save(state, save_path)

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
