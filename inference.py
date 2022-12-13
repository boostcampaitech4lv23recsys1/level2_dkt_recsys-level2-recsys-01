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

from datetime import datetime
from pytz import timezone


def inference_w_one_model(model, data_loader, config, fold):
    model_path = os.path.join(config["trainer"]["save_dir"], config["arch"]["type"])
    model_path = os.path.join(model_path, f"fold_{fold}_best_model.pt")
    state = torch.load(model_path)
    model.load_state_dict(state["state_dict"])
    model.eval()
    device = config["device"]
    predict_list = []

    with torch.no_grad():
        for data in data_loader:
            output = model(data)
            output = output.detach().cpu().numpy()
            predict_list.append(output[:, -1][0])
    return predict_list


def main(config):
    print("---------------------------START PREPROCESSING---------------------------")
    preprocess = Preprocess(config)
    test = preprocess.load_test_data()
    print("---------------------------DONE PREPROCESSING----------------------------")
    test_set = BaseDataset(test, range(len(test)), config)
    test_loader = DataLoader(
        test_set,
        num_workers=config["data_loader"]["args"]["num_workers"],
        shuffle=False,
        batch_size=1,
        collate_fn=collate_fn,
    )

    model = models.get_models(config)

    k = config["preprocess"]["num_fold"]
    final_predict = []
    print("---------------------------START FOLD INFERENCE--------------------------")
    for i in range(k):
        predict = inference_w_one_model(model, test_loader, config, i + 1)
        final_predict.append(predict)
    final_predict = np.array(final_predict)
    final_predict = final_predict.mean(axis=0)
    sample_sub_path = os.path.join(
        config["preprocess"]["data_dir"], "sample_submission.csv"
    )
    sub = pd.read_csv(sample_sub_path)
    sub["prediction"] = final_predict

    sub_path = config["trainer"]["submission_dir"]
    os.makedirs(sub_path, exist_ok=True)
    cur_time = str(datetime.now(timezone("Asia/Seoul")))[:19]
    sub_path = os.path.join(
        sub_path,
        f"inference_{cur_time}_{config['preprocess']['data_ver']}.csv",
    )
    sub.to_csv(sub_path, index=None)
    print("---------------------------DONE PREDICTION-------------------------------")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DKT Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help='config 파일 경로 (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
