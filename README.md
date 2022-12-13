# 1. 프로젝트 개요
## Deep Knowledge Tracing
 DKT는 Deep Knowledge Tracing의 약자로 학생의 '지식 상태'를 추적하는 딥러닝 방법론으로 유저 맞춤화된 교육을 제공하기 위해 중요한 역할을 수행합니다. 이 대회에서는 [Iscream](https://www.i-screamedu.co.kr/index.do) 데이터셋을 이용하여 DKT 모델을 구축하여 학생이 특정 문제를 맞힐지 틀릴지 예측하는 과제를 수행했습니다. 이때 평가 메트릭은 AUROC를 사용했습니다.
 
## 프로젝트 목표
- 코드에 대한 이해를 높이기 위해 베이스라인 코드를 직접 작성하기
- PM을 돌아가면서 맡아 시간 관리 및 소통 능력 배양하기
- 개인이 특정 업무에 종속되는 것이 아니라, 돌아가면서 팀을 맡아 모든 업무를 경험해보기
- 제공된 강의와 자료에서 제시되는 많은 인사이트를 최대한 활용해보기
 
## 활용 장비 및 협업 툴
- GPU: V100 5대
- 운영체제: Ubuntu 18.04.5 LTS
- 협업툴: Github, Notion, Weight & Bias

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

---
# 2. 프로젝트 팀 구성 및 역할
- 프로젝트 전반: **PM** 1인, **전처리** 2인, **베이스라인** 2인 Rotation 방식
- 프로젝트 후반: **PM** 1인, **모델1** 2인, **모델2** 2인 Pair Coding 및 Rotation 방식

---
# 3. 프로젝트 수행 결과 (Private 1위)
<img src="https://user-images.githubusercontent.com/78770033/207284899-abbc901a-69a8-4c69-a596-4a0db1a2a741.png">

---

# 4. References

- Boostcourse 강의 자료
- [Riiid Solution](https://www.kaggle.com/competitions/riiid-test-answer-prediction/discussion/218318)
- [Gated Transformer Networks for Multivariate Time Series Classification](https://arxiv.org/pdf/2103.14438.pdf)

---
# 5. Contributors

| <img src="https://user-images.githubusercontent.com/64895794/200263288-1d77b5f8-ed79-4548-9bc1-01aec2474aaa.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263509-9f564042-6da7-4410-a820-c8198037b0b3.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263683-37597e1d-10c1-483c-90f2-fb4749310e40.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263783-52ddbcf3-5e0b-431e-a84d-f7f17f3d061e.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200264314-77728a99-9849-41e9-b13d-be120877a184.png" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [류명현](https://github.com/ryubright)                                            |                                           [이수경](https://github.com/41ow1ives)                                            |                                            [김은혜](https://github.com/kimeunh3)                                            |                                         [정준환](https://github.com/Jeong-Junhwan)                                          |                                            [장원준](https://github.com/jwj51720)                                            |

