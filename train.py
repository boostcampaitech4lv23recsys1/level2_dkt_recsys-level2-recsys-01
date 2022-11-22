"""
data 불러와서 trainer.py에 넘겨주기
"""


def main(config):
    data = dataloader()
    trainer = trainer(data)
    trainer.train()
