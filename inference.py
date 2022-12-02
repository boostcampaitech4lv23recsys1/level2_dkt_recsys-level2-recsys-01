"""
학습된 모델을 가져와서 submission 생성
"""
from data_loader.preprocess import Preprocess
from data_loader.dataset import BaseDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model as models
from utils import read_json

import torch
import os
import pandas as pd
import numpy as np
import argparse


def inference_w_one_model(model, data_loader, config, fold):
    model_path = os.path.join(config['trainer']['save_dir'], config['arch']['type'])
    model_path = os.path.join(model_path, f'fold_{fold}_best_model.pt')
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])
    model.eval()
    device = config['device']
    predict_list = []
    with torch.no_grad():   
        for data in data_loader:
            output = model(data)
            output = output.detach().cpu().numpy()
            predict_list.append(output[:, -1][0])
    return predict_list
    
def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    if config['arch']['type'] == "LSTM":
        preprocess = Preprocess(config)
        test = preprocess.load_test_data()
        print("---------------------------DONE PREPROCESSING---------------------------")
        
        model = getattr(models, config['arch']['type'])(config).to(config['device'])
        print(f"---------------------------DONE {config['arch']['type']} LOADING---------------------------")
        test_set = BaseDataset(test, range(len(test)), config)
        test_loader = DataLoader(
                    test_set,
                    num_workers=config['data_loader']['args']['num_workers'],
                    shuffle=False,
                    batch_size=1,
                    collate_fn=collate_fn
                    )                
        k = config['preprocess']['num_fold']
        final_predict = []
        print("---------------------------START FOLD INFERENCE---------------------------")
        for i in tqdm(range(k)):
            predict = inference_w_one_model(model, test_loader, config, i+1)
            final_predict.append(predict)
        final_predict = np.array(final_predict)
        final_predict = final_predict.mean(axis=0)
        sample_sub_path = os.path.join(config['preprocess']['data_dir'], 'sample_submission.csv')
        sub = pd.read_csv(sample_sub_path)
        sub['prediction'] = final_predict
        
        sub_path = config['trainer']['submission_dir']
        os.makedirs(sub_path, exist_ok=True)
        sub_path = os.path.join(sub_path, f"inference_{config['preprocess']['data_ver']}.csv")
        sub.to_csv(sub_path, index=None)
        print("---------------------------DONE PREDICTION---------------------------")

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