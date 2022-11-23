import pandas as pd
import numpy as np
import os


class Preprocess:
    def __init__(self, CFG):
        self.config = CFG
        self.train_data = None
        self.test_data = None
    
    ## preprocessing
    def __preprocessing(data):
        raise NotImplementedError
    
    ## feature engineering
    def __feature_engineering(data):
        raise NotImplementedError
    
    ## 데이터 불러오기
    def grouping_data(r, column):
        result = []
        for col in column[1:]:
            result.append(np.array(r[col]))
        return result
        
    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.config.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = df.clolumns.tolist()
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                self.grouping_data, column=columns
                )
            )

        return group.values
    
    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)