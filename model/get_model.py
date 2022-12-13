import model as models


def get_models(config):
    if config["arch"]["type"] == "Transformer":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            dim_model=model_config["dim_model"],
            dim_ffn=model_config["dim_ffn"],
            num_heads=model_config["num_heads"],
            n_layers_transformern_layers_transformer=model_config["n_layers_transformer"],
            dropout_rate=model_config["dropout_rate"],
            embedding_dim=model_config["embedding_dim"],
            device=config["device"],
            config=config,
        ).to(config["device"])

    if config["arch"]["type"] == "GTN":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            dim_model=model_config["dim_model"],
            dim_ffn=model_config["dim_ffn"],
            num_heads=model_config["num_heads"],
            n_layers_transformer=model_config["n_layers_transformer"],
            dropout_rate=model_config["dropout_rate"],
            embedding_dim=model_config["embedding_dim"],
            device=config["device"],
            config=config,
        ).to(config["device"])

    if config["arch"]["type"] == "TransformerLSTM":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            dim_model=model_config["dim_model"],
            dim_ffn=model_config["dim_ffn"],
            num_heads=model_config["num_heads"],
            n_layers_transformer=model_config["n_layers_transformer"],
            n_layers_LSTM=model_config["n_layers_LSTM"],
            dropout_rate=model_config["dropout_rate"],
            embedding_dim=model_config["embedding_dim"],
            device=config["device"],
            config=config,
        ).to(config["device"])

    if config["arch"]["type"] == "LSTM":
        model = getattr(models, config["arch"]["type"])(config).to(config["device"])
        
    if config["arch"]["type"] == "TransformerGRU":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            dim_model=model_config["dim_model"],
            dim_ffn=model_config["dim_ffn"],
            num_heads=model_config["num_heads"],
            n_layers_transformer=model_config["n_layers_transformer"],
            n_layers_LSTM=model_config["n_layers_LSTM"],
            dropout_rate=model_config["dropout_rate"],
            embedding_dim=model_config["embedding_dim"],
            device=config["device"],
            config=config,
        ).to(config["device"]) 
    
    if config["arch"]["type"] == "GRUtransformer":
        model = getattr(models, config["arch"]["type"])(config).to(config["device"])

    if config["arch"]["type"] == "GtnGRU":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            dim_model=model_config["dim_model"],
            dim_ffn=model_config["dim_ffn"],
            num_heads=model_config["num_heads"],
            n_layers_transformer=model_config["n_layers_transformer"],
            n_layers_GRU=model_config["n_layers_GRU"],
            dropout_rate=model_config["dropout_rate"],
            embedding_dim=model_config["embedding_dim"],
            device=config["device"],
            config=config,
        ).to(config["device"])

    return model
