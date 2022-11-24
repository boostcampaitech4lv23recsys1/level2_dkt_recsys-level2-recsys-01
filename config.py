# ====================================================
# CFG
# ====================================================

import os

class CFG:
    use_cuda_if_available = True
    user_wandb = True

    # data
    basepath = "./data"
    data_ver = None
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    model_name = None
    embedding_dim = 64  # int
    num_layers = 1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    data_dir = os.path.join(basepath, data_ver)
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    sceduler = None
    weight_basepath = "./weight"
    optimizer = None
    criterion = None
    metric = None # list