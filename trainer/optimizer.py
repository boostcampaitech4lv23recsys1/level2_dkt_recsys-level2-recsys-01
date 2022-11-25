from torch.optim import Adam, AdamW


def get_optimizer(model, config):
    if config['type'] == "adam":
        optimizer = Adam(model.parameters(), lr=config['args']['lr'], weight_decay=config['args']['weight_decay'])
    if config['type'] == "adamW":
        optimizer = AdamW(model.parameters(), lr=config['args']['lr'], weight_decay=config['args']['weight_decay'])

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer
