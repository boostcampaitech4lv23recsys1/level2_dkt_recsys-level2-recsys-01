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
from utils import read_json, set_seed
from logger import wandb_logger

import wandb
import torch


def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()

    print("-------------------------using cat coloumn list-------------------------")
    print(config["cat_cols"])
    print("-------------------------using num coloumn list-------------------------")
    print(config["num_cols"])
    print("---------------------------DONE PREPROCESSING----------------------------")
    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y-%m-%d_%H:%M")
    
    print("-----------------------------START TRAINING------------------------------")
    if "sweep" in config:
        wandb_train_func = functools.partial(
            run_kfold_sweep, config["preprocess"]["num_fold"], config, data, now
        )
        sweep_config = json.loads(json.dumps(config["sweep"]))
        sweep_id = wandb.sweep(sweep_config, entity=config["entity"], project=config["project"])
        wandb.agent(sweep_id, function=wandb_train_func)
    else:
        run_kfold(config["preprocess"]["num_fold"], config, data, now)
    print("---------------------------DONE TRAINING---------------------------")

def run_kfold_sweep(k, config, data, now):
    kf = KFold(n_splits=k, shuffle=True, random_state=config["trainer"]["seed"])
    wandb.init(name=f'{now}_{config["user"]}_sweep')
    wandb_logger.sweep_update(config, wandb.config)
    val_fold = 0
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(data["userID"].unique().tolist())
    ):
        print(
            f"-------------------------START FOLD {fold + 1} TRAINING---------------------------"
        )
        print(
            f"-------------------------START FOLD {fold + 1} MODEL LOADING----------------------"
        )

        model = models.get_models(config)

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

        result = trainer.train()
        wandb.log(result, step=fold+1)
        val_fold += result['val_aucroc']
        print(
            f"---------------------------DONE FOLD {fold + 1} TRAINING--------------------------"
        )
    wandb.log({"val_fold": val_fold/k})

def run_sweep():
    pass

def run_kfold(k, config, data, now):
    kf = KFold(n_splits=k, shuffle=True, random_state=config["trainer"]["seed"])
    
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(data["userID"].unique().tolist())
    ):
        print(
            f"-------------------------START FOLD {fold + 1} TRAINING---------------------------"
        )
        print(
            f"-------------------------START FOLD {fold + 1} MODEL LOADING----------------------"
        )

        model = models.get_models(config)
        
        wandb.init(project=config["project"], entity=config["entity"], name=f'{now}_{config["user"]}_fold_{fold+1}')
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
    set_seed(config["trainer"]["seed"])

    main(config)
