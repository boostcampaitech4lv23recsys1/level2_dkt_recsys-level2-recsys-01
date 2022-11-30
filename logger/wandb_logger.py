"""
logging을 관리해주는 wandb 함수
"""
import wandb

def init(key, model, config):
    # wandb.login(key)
    wandb.init(project="test-project", entity="dkt-dinosaur")
    # wandb에 기록하고 싶은 정보는 json에서 가져다 update로 추가해줄 수 있다.
    wandb.config = {
            "batch_size" : config["data_loader"]["args"]["batch_size"],
            "epochs": config["trainer"]["epochs"],
            "cat_cols": config["cat_cols"],
            "num_cols": config["num_cols"],
            "optimizer": config["optimizer"]["type"],
        }
    wandb.config.update(config["arch"]["args"])
    wandb.watch(model)
    