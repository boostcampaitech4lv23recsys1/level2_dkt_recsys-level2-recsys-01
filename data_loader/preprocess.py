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

    def __feature_engineering(self, data:pd.DataFrame):
        """
        Selecting features to use.

        Args:
            data (pd.DataFrame): Raw dataframe

        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        data = data[self.feature_engineering]
        
        return data

    ## 공통으로 들어가야 하는 preprocessing (Ex elapsed time : threshold, categories : encoding)
    def __preprocessing(self, data:pd.DataFrame, is_train=True):
        """
        Preprocess for all data needs. For example applying threshold to elapsed time.

        Args:
            data (pd.DataFrame): _description_
            is_train (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: Preprocessed data
        """
        columns = data.columns.tolist()
        data = data.fillna(0)
        train = pd.read_csv(
            f"{self.cfg_preprocess['data_dir']}/train_{self.cfg_preprocess['data_ver']}.csv"
        )
        trainusers = train["userID"].unique()

        # elapsed_time threshold
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

        # value to index for label encoding
        def val2idx(val_list):
            val2idx = {}
            for idx, val in enumerate(val_list):
                val2idx[val] = idx + 1
                
            return val2idx

        for col in self.cat_cols:
            if col != "answerCode":
                tmp2idx = val2idx(data[col].unique().tolist())
                tmp = data[col].map(tmp2idx)
                data.loc[:, f"{col}2idx"] = tmp
                data = data.drop([col], axis=1)

        self.config["cat_col_len"] = {
            col: len(data[f"{col}2idx"]) for col in self.cat_cols
        }

        # split data whether it's train or not
        if is_train:
            data = data[data["answerCode"] != -1].reset_index(drop=True)
        else:
            data = data[~data["userID"].isin(trainusers)].reset_index(drop=True)

        # Transforming features that have periodic property (Ex. Date, Dat, Month ...)
        def process_periodic(data: pd.Series, period: int, process_type: str = "sin"):
            if process_type == "sin":
                return np.sin(2 * np.pi / period * data)
            if process_type == "cos":
                return np.cos(2 * np.pi / period * data)
        if "test_hour" in columns:
            data["test_hour"] = process_periodic(data=data["test_hour"], period=24)
            
        return data

    def __data_augmentation(self, data: pd.DataFrame):
        """
        Splitting user as new user if the user has solved number of problems more than max_seq_len.
        
        Args:
            data (pd.DataFrame): Raw dataframe

        Returns:
            pd.DataFrame: dataframe
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
        
        return self.test_data

    def gkf_data(self, data):
        k = self.cfg_preprocess["num_fold"]
        gkf = GroupKFold(n_splits=k)
        group = data["userID"].tolist()

        return gkf, group

