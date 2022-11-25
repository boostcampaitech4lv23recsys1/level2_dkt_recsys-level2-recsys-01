"""
data 불러와서 trainer.py에 넘겨주기
"""
from data_loader.dataset import XGBoostDataset
from trainer.trainer import XGBoostTrainer
from model.XGBoost import XGBoost
from utils.util import FEATURES
from config import CFG


def main(config):
    if config.model_name == "XGBoost":
        data = XGBoostDataset(config)
        train, valid = data._split()
        train_y = train["answerCode"]
        train_X = train.drop(["answerCode"], axis=1)
        valid_y = valid["answerCode"]
        valid_X = valid.drop(["answerCode"], axis=1)

        model = XGBoost().model

        trainer = XGBoostTrainer(
            model=model,
            train_X=train_X,
            train_y=train_y,
            test_X=valid_X,
            test_y=valid_y,
            features=FEATURES,
            early_stopping_rounds=100,
            verbose=5,
        )
        trainer.train()


if __name__ == "__main__":
    main(CFG)