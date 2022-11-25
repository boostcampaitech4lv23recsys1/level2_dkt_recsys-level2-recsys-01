# ====================================================
# CFG
# ====================================================

import os

class CFG:
    use_cuda_if_available = True
    user_wandb = True

    # data
    basepath = "./data"
    data_ver = "traintest_v1.2.csv"
    test_file = "test_data.csv"
    loader_verbose = True
    cat_cols = [
        'userID',
        'assessmentItemID',
        'testId',
        'week_num',
        'KnowledgeTag',
        'test_cat',
        'question_number',
        'question_numslen',
        'test_month',
        'test_day',
        'test_hour',
        'exp_test',
        'exp_tag',
        'exp_question',
    ]

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    model_name = "XGBoost"
    embedding_dim = 64  # int
    num_layers = 1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    data_path = os.path.join(basepath, data_ver)
    test_path = os.path.join(basepath, test_file)
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    sceduler = None
    weight_basepath = "./weight"
    optimizer = None
    criterion = None
    metric = None