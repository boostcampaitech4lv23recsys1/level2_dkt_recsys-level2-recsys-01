import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_preprocess = config["preprocess"]

        self.cat_cols = config["cat_cols"]
        self.num_cols = config["num_cols"]
        self.augmentation = self.cfg_preprocess["data_augmentation"]

        self.feature_engineering = (
            ["userID", "answerCode", "testId"] + self.cat_cols + self.num_cols
        )

        self.train_data = None
        self.test_data = None

    ## feature engineering
    def __feature_engineering(self, data):
        data = data[self.feature_engineering]
        return data

    ## 공통으로 들어가야 하는 preprocessing (Ex elapsed time : threshold, categories : encoding)
    def __preprocessing(self, data, is_train=True):
        columns = data.columns.tolist()
        data = data.fillna(0)
        train = pd.read_csv(
            f"{self.cfg_preprocess['data_dir']}/train_{self.cfg_preprocess['data_ver']}.csv"
        )
        trainusers = train["userID"].unique()

        # elapsed_time
        if "elapsed_time" in columns:
            threshold, imputation = self.cfg_preprocess["fe_elapsed_time"]
            data.loc[
                data[data["elapsed_time"] > threshold].index, "elapsed_time"
            ] = threshold
            if imputation != "cut":
                if imputation == "median":
                    test2time = data.groupby("testId")["elapsed_time"].median().to_dict()
                elif imputation == "mean":
                    test2time = data.groupby("testId")["elapsed_time"].mean().to_dict()
                    
                data.loc[data[data["elapsed_time"] == threshold].index, "elapsed_time"] = data[
                    data["elapsed_time"] == threshold
                ]["testId"]
                data.loc[
                    data[data["elapsed_time"].apply(type) == str].index, "elapsed_time"
                ] = data[data["elapsed_time"].apply(type) == str]["elapsed_time"].map(
                    test2time
                )

        def val2idx(val_list):
            val2idx = {}
            for idx, val in enumerate(val_list):
                val2idx[val] = idx + 1
            return val2idx

        # 어차피 cat_cols일 경우만 돌아가는 거라면 columns가 아니라 cat_cols를 돌아도 되는거 아닌가?
        for col in columns:
            if col in self.cat_cols:
                if col != "answerCode":
                    tmp2idx = val2idx(data[col].unique().tolist())
                    tmp = data[col].map(tmp2idx)
                    data.loc[:, f"{col}2idx"] = tmp
                    data = data.drop([col], axis=1)

        self.config["cat_col_len"] = {
            col: len(data[f"{col}2idx"]) for col in self.cat_cols
        }

        # data['userID'] = val2idx(data['userID'].unique().tolist()) # 얘를 해야함

        if is_train:
            data = data[data["answerCode"] != -1].reset_index(drop=True)
        else:
            # if self.augmentation:
            #     data = data[~data["userID_origin"].isin(trainusers)].reset_index(
            #         drop=True
            #     )
            # else:
            data = data[~data["userID"].isin(trainusers)].reset_index(drop=True)
        
        

        # 주기적인 성질(날짜, 요일, 월 등)을 갖는 column을 sin, cos을 이용해서 변환하기
        def process_periodic(data: pd.Series, period: int, process_type: str = "sin"):
            if process_type == "sin":
                return np.sin(2 * np.pi / period * data)
            if process_type == "cos":
                return np.cos(2 * np.pi / period * data)

        # 사용방법 예시
        # data["주기적 성질을 갖는 컬럼"] = process_periodic(data=data["주기적 성질을 갖는 컬럼"], period=주기)
        if "test_hour" in columns:
            data["test_hour"] = process_periodic(data=data["test_hour"], period=24)

        if "test_month" in columns:
            data["test_month"] = process_periodic(data=data["test_month"], period=12)
            
        return data

    def load_data_from_file(self):
        df = pd.read_csv(
            f"{self.cfg_preprocess['data_dir']}/traintest_{self.cfg_preprocess['data_ver']}.csv"
        )
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        return df

    def load_train_data(self):
        self.train_data = self.load_data_from_file()
        self.train_data = self.__feature_engineering(self.train_data)
        self.train_data = self.__preprocessing(self.train_data, is_train=True)
        if self.augmentation:
            self.train_data = self.__data_augmentation(self.train_data)
        return self.train_data

    def load_test_data(self):
        self.test_data = self.load_data_from_file()
        self.test_data = self.__feature_engineering(self.test_data)
        self.test_data = self.__preprocessing(self.test_data, is_train=False)
        # test에는 augmentation이 없는게 맞다?
        return self.test_data

    def gkf_data(self, data):
        k = self.cfg_preprocess["num_fold"]
        gkf = GroupKFold(n_splits=k)
        group = data["userID"].tolist()

        return gkf, group

    def __data_augmentation(self, data: pd.DataFrame):
        """
        유저가 푼 문제를 max_seq_len에 따라 새로운 유저로 분리
        유저의 최댓값이 7441이라 그냥 10000을 곱해서 아예 새로운 유저를 만들어줬음

        즉, 새로 나온 유저의 id를 기존의 아이디로 바꾸고 싶으면
        10000으로 나눴을 때의 몫을 취하면 됨.
        """
        max_seq_len = self.config["dataset"]["max_seq_len"]
        group = data.groupby("userID").get_group
        unique_user = data["userID"].unique()

        whole_new_user_id = []
        for user in unique_user:
            temp_new_user_id = np.zeros_like(group(user)["userID"])
            added_to_user = 10000 * user
            temp_new_user_id += [
                x // max_seq_len + added_to_user for x in range(len(temp_new_user_id))
            ]
            whole_new_user_id.extend(temp_new_user_id)

        data["userID_origin"] = data["userID"].copy()
        encoder = LabelEncoder()
        data["userID"] = encoder.fit_transform(whole_new_user_id)
        
        return data
