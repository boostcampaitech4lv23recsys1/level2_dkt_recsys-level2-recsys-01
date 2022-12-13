import os
import torch
from numpy import inf
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from . import loss, metric, optimizer, scheduler
from logger import wandb_logger
from utils import MetricTracker
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        config: dict,
        fold: int,
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config
        self.cfg_trainer = config["trainer"]
        self.fold = fold

        self.device = config["device"]

        self.criterion = loss.get_loss(config)
        self.metric_ftns = self.cfg_trainer["metric"]
        self.optimizer = optimizer.get_optimizer(self.model, config["optimizer"])
        self.lr_scheduler = scheduler.get_scheduler(self.optimizer, config)

        self.train_metrics = MetricTracker("loss", *self.metric_ftns)
        self.valid_metrics = MetricTracker("loss", *self.metric_ftns)

        self.epochs = self.cfg_trainer["epochs"]
        self.start_epoch = 1

        self.save_dir = self.cfg_trainer["save_dir"]
        self.min_val_loss = inf
        self.max_val_aucroc = 0
        self.model_name = type(self.model).__name__

    def _train_epoch(self):
        log = dict()
        total_outputs = []
        total_targets = []

        self.model.train()
        self.train_metrics.reset()
        print("...Train...")
        for data in tqdm(self.train_data_loader):
            target = data["answerCode"].to(self.device)
            output = self.model(data)
            loss = self._compute_loss(output, target)
            self.train_metrics.update("loss", loss.item())

            output = output[:, -1]
            target = target[:, -1]

            total_outputs.append(output.detach())
            total_targets.append(target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for met in self.metric_ftns:
            ftns = metric.get_metric(met)
            output_to_cpu = torch.cat(total_outputs).cpu().numpy()
            target_to_cpu = torch.cat(total_targets).cpu().numpy()

            self.train_metrics.update(met, ftns(output_to_cpu, target_to_cpu))
        train_log = self.train_metrics.result()
        log.update(**{f"train_{k}": v for k, v in train_log.items()})

        total_outputs = []
        total_targets = []
        self.model.eval()
        self.valid_metrics.reset()
        print("...Valid...")
        for data in tqdm(self.valid_data_loader):
            target = data["answerCode"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self._compute_loss(output, target)
            self.valid_metrics.update("loss", loss.item())

            output = output[:, -1]
            target = target[:, -1]

            total_outputs.append(output.detach())
            total_targets.append(target.detach())

        for met in self.metric_ftns:
            ftns = metric.get_metric(met)
            output_to_cpu = torch.cat(total_outputs).cpu().numpy()
            target_to_cpu = torch.cat(total_targets).cpu().numpy()
            self.valid_metrics.update(met, ftns(output_to_cpu, target_to_cpu))
        val_log = self.valid_metrics.result()
        log.update(**{f"val_{k}": v for k, v in val_log.items()})

        return log

    def train(self):
        best_result = {}

        for epoch in range(self.start_epoch, self.epochs + 1):
            print(
                f"-----------------------------EPOCH {epoch} TRAINING----------------------------"
            )
            result = self._train_epoch()
            if "sweep" not in self.config:
                wandb.log(result, step=epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step(result["val_aucroc"])

            if result["val_aucroc"] > self.max_val_aucroc:
                self.state = {
                    "model_name": self.model_name,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                }
                self.max_val_aucroc = result["val_aucroc"]
                best_result = result
        if "sweep" not in self.config:
            self._save_checkpoint()
        else:
            return best_result

    def _save_checkpoint(self):
        print("...SAVING MODEL...")

        save_path = os.path.join(self.save_dir, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"fold_{self.fold}_best_model.pt")
        torch.save(self.state, save_path)

    def _compute_loss(self, output, target):
        loss = self.criterion(output, target)

        return loss


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
            verbose=self.verbose,
        )

        preds = self.model.predict_proba(self.test_X[self.features])[:, 1]
        acc = accuracy_score(self.test_y, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(self.test_y, preds)

        print(f"VALID AUC : {auc} ACC : {acc}\n")
