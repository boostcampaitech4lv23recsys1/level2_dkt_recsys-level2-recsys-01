from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, config):
    if 'lr_scheduler' in config:
        if config['lr_scheduler']["type"] == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                patience=config['lr_scheduler']["patience"],
                factor=config['lr_scheduler']["factor"],
                mode=config['lr_scheduler']["mode"],
                verbose=True
            )
            
        if config['lr_scheduler'] == "linear_warmup":
            warmup_steps = 0 # len(dataloader) * 2
            total_steps = 0 # len(dataloader) * epochs
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
    return None
