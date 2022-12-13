import numpy as np
import pandas as pd
import os


class Ensemble:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filenames = os.listdir(filepath)
        self.output_list = []

        output_path = [filepath + filename for filename in self.filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop("prediction", axis=1)
        self.output_df = self.output_frame.copy()
        self.csv_nums = len(output_path)

        for path in output_path:
            self.output_list.append(pd.read_csv(path)["prediction"].to_list())
        for filename, output in zip(self.filenames, self.output_list):
            self.output_df[filename] = output

    def simple_weighted(self, weight: list):
        """
        Ensembles with manually designated weight

        Args:
            weight (list): list of weight

        Raises:
            ValueError: Raise error when length of model and weight is not identical.
            ValueError: Sum of input weights have to be 1

        Returns:
            list: List of result.
        """
        if not len(self.output_list) == len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight) != 1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

        pred_arr = np.append([self.output_list[0]], [self.output_list[1]], axis=0)
        for i in range(2, len(self.output_list)):
            pred_arr = np.append(pred_arr, [self.output_list[i]], axis=0)
        result = np.dot(pred_arr.T, np.array(weight))
        return result.tolist()

    def average_weighted(self):
        """
        Ensembles with 1/n weight.

        Returns:
            list: List of result.
        """
        weight = [1 / len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [
            pred * np.array(w) for pred, w in zip(self.output_list, weight)
        ]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()

    def mixed(self):
        """
        If negative case causes, ensemble with next predicted rating.

        Returns:
            list: List of result.
        """
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames) - 1):
            pre_idx = self.filenames[idx]
            post_idx = self.filenames[idx + 1]
            result[self.output_df[pre_idx] < 1] = self.output_df.loc[
                self.output_df[pre_idx] < 1, post_idx
            ]
        return result.tolist()
    
    def hardsoft(self):
        """
        ensemble hard first, soft second
        
        Example: 
            problem 1: [0.1, 0.1, 0.6, 0.6, 0.8]
                soft(weighted sum) -> 2.2/5 = 0.44
                hardsoft -> over 0.5 (3), under 0.5 (2) -> soft ensemble [0.6, 0.6, 0.8] -> 2.0/3 = 0.66667
        """
        output = []
        self.output_df = self.output_df.values[:,1:]
        for row in self.output_df:
            voting = np.where(row > 0.5,1,0)
            if voting.sum() > self.csv_nums//2:
                value = np.average(row[np.where(voting == 1)])
            elif voting.sum() < self.csv_nums//2:
                value = np.average(row[np.where(voting == 0)])
            else:
                value = np.average(row)
            output.append(value)
        result = pd.Series(output)
        return result.tolist()
        
