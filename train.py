from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset
# from config import CFG

"""
data 불러와서 trainer.py에 넘겨주기
"""

def main(config):
    preprocess = Preprocess
    data = preprocess.load_train_data()
    gkf, group = preprocess.gkf_data(data)
    for train_idx, val_idx in gkf.split(data, groups=group):
        train_set = BaseDataset(data, train_idx)
        val_set = BaseDataset(data, val_idx)
    
        # trainer = trainer(data)a
        # trainer.train()
