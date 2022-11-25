# ====================================================
# CFG
# ====================================================

class CFG:
    use_cuda_if_available = True
    user_wandb = True

    # data
    basepath = "./data"
    data_ver = 'v1.1'
    loader_verbose = True

    # dump
    output_dir = "./submission/"
    pred_file = "submission.csv"

    # build
    model_name = None
    embedding_dim = 64  # int
    num_layers = 1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    data_dir = f'{basepath}/traintest_{data_ver}.csv'
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    max_seq_len = 20
    sceduler = None
    weight_basepath = "./weight"
    optimizer = None
    criterion = None
    weight_basepath = "./saved/models/"

    metric = None
    feature_engineering = ['userID', 'assessmentItemID', 'testId', 'answerCode',
                           'week_num', 'elapsed_time', 'time_question_median',
                            'time_user_median', 'time_user_mean', 'KnowledgeTag', 'test_cat',
                            'test_id', 'question_number', 'question_numslen', 'test_month',
                            'test_day', 'test_hour', 'user_acc', 'test_acc', 'tag_acc',
                            'question_acc', 'month_acc', 'hour_acc', 'exp_test', 'exp_tag',
                            'exp_question', 'ans_cumsum', 'continuous_user', 'continuous_test'] # 쓸 column들을 남겨주세요
    fe_elapsed_time = [200, 'mean'] # elapsed time의 이상치 threshold와 -1 imputation 방법
    k_fold = [True, 5]