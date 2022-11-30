"""
학습된 모델을 가져와서 submission 생성
"""
from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset
from torch.utils.data import DataLoader

import model as models
from utils import read_json

import torch
import os
import pandas as pd
import numpy as np
import argparse


def inference_w_one_model(model, data_loader, config):
    model.load_state_dict('모델 weight 경로 by config')
    model.eval()
    device = config['device']
    predict_list = []
    with torch.no_grad():   
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            output = output.detach().cpu().numpy()
            predict_list.append(output)
    return predict_list
    
def main(config):
    if config.model_name == "LSTM":
        preprocess = Preprocess(config)
        test = preprocess.load_test_data()
        
        model_args = config['arch']['args']
        model_args.update({
        'cat_cols': config['cat_cols'],
        'num_cols': config['num_cols'], 
        })
        model = getattr(models, config['arch']['type'])(model_args)
        test_set = BaseDataset(test, range(len(test)), config)
        test_loader = DataLoader(
                    test_set,
                    num_workers=config['num_workers'],
                    shuffle=False,
                    batch_size=1
                    )                
        k = config['kfold 지정한 fold 개수']
        final_predict = []
        for i in range(k):
            predict = inference_w_one_model(model, test_loader, config)
            final_predict.append(predict)
        final_predict = np.array(final_predict)
        final_predict = final_predict.mean(axis=0)
        
        sub = pd.read_csv('config[sample_submission이 있는 경로]')
        sub['prediction'] = final_predict
        sub.to_csv(f'./data/inference_{"데이터버전적어주세용"}') 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DKT Dinosaur')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config 파일 경로 (default: "./config.json")')
    args = args.parse_args()
    config = read_json(args.config)
    
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
    
    #     if config.model_name == "XGBoost":
    #     test = data.train[data.train["userID"].isin(data.test_ids)]

    #     test = test[test["userID"] != test["userID"].shift(-1)]
    #     test = test.drop(["answerCode"], axis=1)

    #     total_preds = model.predict_proba(test[FEATURES])[:, 1]

    #     output_dir = "output/"
    #     write_path = os.path.join(output_dir, "submission.csv")
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     with open(write_path, "w", encoding="utf8") as w:
    #         print("writing prediction : {}".format(write_path))
    #         w.write("id,prediction\n")
    #         for id, p in enumerate(total_preds):
    #             w.write("{},{}\n".format(id, p))