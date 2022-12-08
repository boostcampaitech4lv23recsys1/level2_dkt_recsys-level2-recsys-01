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
├── ensemble.py - make ensemble with submission files
│
├── config/ - holds configurations for training
|   ├──LSTM_config.json
|   ├──transformer_config.json
|   ├──transformerLSTM_config.json
|   ├──GRUtransformer_config.json
│   └──GTN_congfig.json
│
├── data_loader/ - anything about data loading goes here
│   ├── dataset.py
│   └── preprocess.py
│
├── data/ - default directory for storing input data
│
├── model/ - base, get_model, utils for model, and all of models
│   ├── base.py
│   ├── get_model.py
│   ├── utils.py
│   ├── LSTM.py
│   ├── transformer.py
│   ├── transformerLSTM.py
│   ├── transformerGRU.py
│   ├── GRUtransformer.py
│   ├── GTN.py
│   ├── GTNGRU.py
│   └── XGBoost.py
│
├── trainer/ - trainers, losses, metric, optimizer, and scheduler
│   ├── trainer.py
│   ├── loss.py
│   ├── metric.py
│   ├── optimizer.py
│   └── scheduler.py
|
├── preprocess/ - preprocess ipynb files
│
├── ensembles/ - anything about ensemble goes here
│   └── ensemble.py
|
├── ensembles_inference/ - submission files that needs to be ensembled
|
├── logger/ - module for wandb  and logging
│   └── wandb_logger.py
│
├── saved_model/
|
├── submission/
|
└── utils/ - small utility functions
    ├── util.py
    └── ...
```

## Reference

이것 저것

## Contributors

팀원 소개
