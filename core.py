import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from model.elastic_net import ElasticNetSearcher
from utils.logger import init_logger


class BaselinePipeline:
    def __init__(self):
        self.logger = init_logger()

    def process(self):
        """
            1. data_pipeline:
                    load dataset
            2. feature_extraction_pipeline: 
                    extract features and preprocess data
            3. model_pipeline: 
                    train model and estimate metric
        :return: exit code
        """
        try:
            processed = self.feature_extraction_pipeline()
            metric = self.model_pipeline()
        except ImportError as e:
            # TODO: handling ImportError
            self.logger.exception(e)
            return False
        except Exception as e:
            self.logger.exception(e)
            return False

        return True

    @property
    def dataset(self):
        return self.data_pipeline()

    @property
    def columns(self):
        # header of dataset
        return self.dataset.columns

    def data_pipeline(self):
        return self.load_dataset()

    def feature_extraction_pipeline(self):
        return self.dataset

    def model_pipeline(self):
        """
            1. build dataset
            2. train model ( + gridsearch)
            3. estimate metrics
        :return: metric
        """
        # build dataset
        x_train, y_train, x_test, y_test = self.build_dataset()

        # train model
        searcher = ElasticNetSearcher(
            x_train=x_train, y_train=y_train,
            score=mean_absolute_error,
            grid_params={
                "max_iter": [1, 5, 10],
                "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)
            }
        )
        searcher.fit()

        # estimate metrics
        pred_y = searcher.predict(X=x_test)
        metric = searcher.estimate_metric(y_true=y_test, y_pred=pred_y)
        searcher.save(prefix="result/baseline/")
        return metric

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        # read DataFrame from xls file
        return pd.read_excel(
            'origin/Concrete_Data.xls',
            sheet_name='Sheet1'
        )

    def build_dataset(self):
        """
            split Dataset
        :return: train Xy, test Xy
        """
        # split train test
        # if you change random_state, u get a result slightly different from report
        train, test = train_test_split(self.dataset, shuffle=True, random_state=0)

        # split X y
        train_x, train_y = self.split_xy(train)
        test_x, test_y = self.split_xy(test)

        return train_x, train_y, test_x, test_y

    def split_xy(self, df: pd.DataFrame):
        return df.drop(columns=self.columns[-1]), df[self.columns[-1]]
