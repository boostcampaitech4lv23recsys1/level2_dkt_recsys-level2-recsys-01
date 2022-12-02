"""
logging을 관리해주는 wandb 함수
"""
import wandb

def init(now, config, fold):
    wandb.init(name=f'{now}_{config["user"]}_fold_{fold}')
    # wandb에 기록하고 싶은 정보는 json에서 가져다 update로 추가해줄 수 있다.
    wandb.config = {
            "batch_size" : config["data_loader"]["args"]["batch_size"],
            "epochs": config["trainer"]["epochs"],
            "fe_elapsed_time": config["preprocess"]["fe_elapsed_time"][0],
            "max_seq_len": config["dataset"]["max_seq_len"], 
            "optimizer": config["optimizer"]["type"],
            "learning_rate": config["optimizer"]["args"]["lr"],
            "weight_decay": config["optimizer"]["args"]["weight_decay"],
        }
    wandb.config.update(config["arch"]["args"])
    
    return wandb.config
    
def sweep_update(config, w_config):
    config["data_loader"]["args"]["batch_size"] = w_config["batch_size"]
    config["trainer"]["epochs"] = w_config["epochs"]
    config["preprocess"]["fe_elapsed_time"][0] = w_config["fe_elapsed_time"]
    config["dataset"]["max_seq_len"] = w_config["max_seq_len"]
    config["optimizer"]["type"] = w_config["optimizer"]
    config["optimizer"]["args"]["lr"] = w_config["learning_rate"]
    config["optimizer"]["args"]["weight_decay"] = w_config["weight_decay"]

    config["arch"]["args"] = { k: w_config[k] for k, _ in config["arch"]["args"].items() }