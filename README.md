## Deep Knowledge Tracing

프로젝트 개요 설명

## Requirements

```
python
pytorch
등등
```

## Folder Structure

```
Deep Knowledge Tracing/
│
├── train.py - main script to start training
├── inference.py - make submission with trained models
│
├── config.py - holds configuration for training
│
├── data_loader/ - anything about data loading goes here
│   ├── data_loaders.py
│   ├── dataset.py
│   └── preprocess.py
│
├── data/ - default directory for storing input data
│
├── model/ - models
│   ├── model_example_1.py
│   └── model_example_2.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for wandb and logging output
│
├── trainer/ - trainers, losses, metric, optimizer, and scheduler
│   ├── trainer.py
│   ├── loss.py
│   ├── metric.py
│   ├── optimizer.py
│   └── scheduler.py
│
├── logger/ - module for wandb  and logging
│   └── wandb_logger.py
│
└── utils/ - small utility functions
    ├── util.py
    └── ...
```

## Reference

이것 저것

## Contributors

팀원 소개
