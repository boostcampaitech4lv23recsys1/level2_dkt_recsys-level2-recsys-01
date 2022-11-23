"""
logging을 관리해주는 wandb 함수
"""
import wandb
from config import CFG

def init(key, model):
    wandb.login(key)
    wandb.init(project='dkt-dinosaur')
    wandb.config = (vars(CFG))
    wandb.watch(model)
    