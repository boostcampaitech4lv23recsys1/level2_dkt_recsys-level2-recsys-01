def sweep_update(config, w_config):
    config["data_loader"]["args"]["batch_size"] = w_config["batch_size"]
    config["trainer"]["epochs"] = w_config["epochs"]
    config["dataset"]["max_seq_len"] = w_config["max_seq_len"]
    config["optimizer"]["type"] = w_config["optimizer"]
    config["optimizer"]["args"]["lr"] = w_config["learning_rate"]
    config["optimizer"]["args"]["weight_decay"] = w_config["weight_decay"]

    config["arch"]["args"] = { k: w_config[k] for k, _ in config["arch"]["args"].items() }