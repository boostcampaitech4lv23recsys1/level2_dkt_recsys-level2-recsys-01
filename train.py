import argparse

from trainer import BaseTrainer
from utils import read_json
import model as models

import wandb
import torch
from torch.utils.data.dataloader import default_collate
from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset, get_loader, collate

"""
data 불러와서 trainer.py에 넘겨주기
"""


def main(config):
    model = getattr(models, config['arch']['type'])
    preprocess = Preprocess
    data = preprocess.load_train_data()
    gkf, group = preprocess.gkf_data(data)
    for train_idx, val_idx in gkf.split(data, groups=group):
        train_set = BaseDataset(data, train_idx)
        val_set = BaseDataset(data, val_idx)
        train, valid = get_loader(
            train_set, val_set, collate=default_collate
        )
    
        trainer = BaseTrainer(
            model=model,
            train_data_loader=train,
            valid_data_loader=valid,
            config=config,
        )

        if config['sweep']:
            sweep_id = wandb.sweep(config['sweep'])
            wandb.agent(sweep_id, trainer.train, count=1)
        else:
            trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DKT Dinosaur')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config 파일 경로 (default: "./config.json")')
    args = args.parse_args()
    config = read_json(args.config)
    
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
