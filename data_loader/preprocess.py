import pandas as pd
from sklearn.model_selection import GroupKFold


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_preprocess = config['preprocess']

        self.cat_cols = config['cat_cols']
        self.num_cols = config['num_cols']

        self.feature_engineering = ['userID'] + self.cat_cols + self.num_cols

        self.train_data = None
        self.test_data = None

    ## feature engineering
    def __feature_engineering(self, data):
        data = data[self.feature_engineering]
        return data

    ## 공통으로 들어가야 하는 preprocessing (Ex elapsed time : threshold, categories : encoding)
    def __preprocessing(self, data, is_train=True):
    
        columns = data.columns.tolist()
        train = pd.read_csv(f"{self.cfg_preprocess['data_dir']}/train_{self.cfg_preprocess['data_ver']}.csv")
        trainusers = train['userID'].unique()
        
        # elapsed_time
        if "elapsed_time" in columns:
            threshold, imputation = self.cfg_preprocess['fe_elapsed_time']
            data.loc[
                data[data["elapsed_time"] > threshold].index, "elapsed_time"
            ] = threshold
            if imputation == "median":
                test2time = data.groupby("testId")["elapsed_time"].median().to_dict()
            elif imputation == "mean":
                test2time = data.groupby("testId")["elapsed_time"].mean().to_dict()
            data.loc[data[data["elapsed_time"] == -1].index, "elapsed_time"] = data[
                data["elapsed_time"] == -1
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

        for col in columns:
            if col in self.cat_cols:
                if col!='answerCode':
                    tmp2idx = val2idx(data[col].unique().tolist())
                    tmp = data[col].map(tmp2idx)
                    data.loc[:, f"{col}2idx"] = tmp
                    data = data.drop([col], axis=1)
    
        # data['userID'] = val2idx(data['userID'].unique().tolist()) # 얘를 해야함
        
        if is_train:
            data = data[data["answerCode"]!=-1].reset_index(drop=True)
        else:
            data = data[~data['userID'].isin(trainusers)].reset_index(drop=True)

        return data

    def load_data_from_file(self):
        df = pd.read_csv(f"{self.cfg_preprocess['data_dir']}/traintest_{self.cfg_preprocess['data_ver']}.csv")
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        return df

    def load_train_data(self):
        self.train_data = self.load_data_from_file()
        self.train_data = self.__feature_engineering(self.train_data)
        self.train_data = self.__preprocessing(self.train_data, is_train=True)
        return self.train_data

    def load_test_data(self):
        self.test_data = self.load_data_from_file()
        self.test_data = self.__feature_engineering(self.test_data)
        self.test_data = self.__preprocessing(self.test_data, is_train=False)
        return self.test_data

    def gkf_data(self, data):
        k = self.cfg_preprocess['num_fold']
        gkf = GroupKFold(n_splits=k)
        group = data["userID"].tolist()
        return gkf, group
