import argparse

from trainer import BaseTrainer
from utils import read_json
import model as models

import wandb
import torch

"""
data 불러와서 trainer.py에 넘겨주기
"""


def main(config):
    train, valid = dataloader()
    model = getattr(models, config['arch']['type'])
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