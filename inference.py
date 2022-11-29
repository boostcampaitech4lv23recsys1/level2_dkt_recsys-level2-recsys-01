"""
학습된 모델을 가져와서 submission 생성
"""
from utils.util import FEATURES
import os

def inference(config, data, model):
    if config.model_name == "XGBoost":
        test = data.train[data.train["userID"].isin(data.test_ids)]

        test = test[test["userID"] != test["userID"].shift(-1)]
        test = test.drop(["answerCode"], axis=1)

        total_preds = model.predict_proba(test[FEATURES])[:, 1]

        output_dir = "output/"
        write_path = os.path.join(output_dir, "submission.csv")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(write_path, "w", encoding="utf8") as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))
