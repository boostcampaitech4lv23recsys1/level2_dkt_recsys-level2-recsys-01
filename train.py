import argparse
import functools
from pytz import timezone
from datetime import datetime

import model as models
from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset, get_loader
from sklearn.model_selection import KFold
from trainer import BaseTrainer
from utils import read_json

import wandb
import torch

"""
data 불러와서 trainer.py에 넘겨주기
"""
from data_loader.dataset import XGBoostDataset
from trainer.trainer import XGBoostTrainer
from model.XGBoost import XGBoost
from utils.util import FEATURES
from config import CFG
from logger import wandb_logger

def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()
    print("---------------------------DONE PREPROCESSING----------------------------")
    print("---------------------------START MODEL LOADING---------------------------")
    
    model = getattr(models, config['arch']['type'])(config).to(config['device'])
    print("---------------------------DONE MODEL LOADING----------------------------")
    wandb_train_func = functools.partial( \
        run_kfold, \
        config['preprocess']['num_fold'], \
        config, \
        model, \
        data)
    print("-----------------------------START TRAINING------------------------------")
    if 'sweep' in config:
        sweep_id = wandb.sweep(config['sweep'])
        wandb.agent(sweep_id, wandb_train_func, count=1)
    else:
        wandb_train_func()
    print("---------------------------DONE TRAINING---------------------------")

def run_kfold(k, config, model, data):
    kf = KFold(n_splits=k)
    now = datetime.now(timezone('Asia/Seoul')).strftime(f'%Y-%m-%d_%H:%M')
    for fold, (train_idx, val_idx) in enumerate(kf.split(data['userID'].unique().tolist())):
        print(f"--------------------------START FOLD {fold+1} TRAINING--------------------------") 
        wandb_logger.init(now, model, config, fold+1)

        train_set = BaseDataset(data, train_idx, config)
        val_set = BaseDataset(data, val_idx, config)
        
        train, valid = get_loader(train_set, val_set, config['data_loader']['args'])
        
        trainer = BaseTrainer(
            model=model,
            train_data_loader=train,
            valid_data_loader=valid,
            config=config,
            fold=fold+1
        )

        trainer.train()
        print(f"---------------------------DONE FOLD {fold+1} TRAINING--------------------------")
        wandb.finish()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DKT Dinosaur')
    args.add_argument('-c', '--config', default='./LSTM_Test.json', type=str,
                      help='config 파일 경로 (default: "./config.json")')
    args = args.parse_args()
    config = read_json(args.config)
    
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)