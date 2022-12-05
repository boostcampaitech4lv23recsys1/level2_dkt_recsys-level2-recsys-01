import argparse
import functools
from pytz import timezone
from datetime import datetime
import json

import model as models
from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset, get_loader
from sklearn.model_selection import KFold
from trainer import BaseTrainer
from utils import read_json
from logger import wandb_logger

import wandb
import torch

"""
data 불러와서 trainer.py에 넘겨주기
"""


def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()

    print("---------------------------DONE PREPROCESSING----------------------------")
    wandb_train_func = functools.partial(
        run_kfold, config["preprocess"]["num_fold"], config, data
    )
    print("-----------------------------START TRAINING------------------------------")
    if "sweep" in config:
        # breakpoint()
        sweep_config = json.loads(json.dumps(config["sweep"]))
        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, function=wandb_train_func, count=1)
    else:
        wandb_train_func()
    print("---------------------------DONE TRAINING---------------------------")


def run_kfold(k, config, data):
    kf = KFold(n_splits=k)

    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y-%m-%d_%H:%M")
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(data["userID"].unique().tolist())
    ):
        print(
            f"-------------------------START FOLD {fold + 1} TRAINING---------------------------"
        )
        print(
            f"-------------------------START FOLD {fold + 1} MODEL LOADING----------------------"
        )
        w_config = wandb_logger.init(now, config, fold+1)

        # config update for sweep
        if "sweep" in config:
            wandb_logger.sweep_update(config, w_config)

        model = models.get_models(config)
        
        wandb.watch(model)

        print(
            f"-------------------------DONE FOLD {fold + 1} MODEL LOADING-----------------------"
        )
        
        train_set = BaseDataset(data, train_idx, config)
        val_set = BaseDataset(data, val_idx, config)

        train, valid = get_loader(train_set, val_set, config["data_loader"]["args"])

        trainer = BaseTrainer(
            model=model,
            train_data_loader=train,
            valid_data_loader=valid,
            config=config,
            fold=fold + 1,
        )

        trainer.train()
        print(
            f"---------------------------DONE FOLD {fold + 1} TRAINING--------------------------"
        )
        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DKT Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="./LSTM_Test.json",
        type=str,
        help='config 파일 경로 (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
