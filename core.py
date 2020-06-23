import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from feature_extractor.core import FeatureExtractor
from model.elastic_net import ElasticNetSearcher
from utils.logger import init_logger


class BaselinePipeline:
    def __init__(self, p_type):
        self.logger = init_logger()
        self.p_type = p_type

    @property
    def dataset(self) -> pd.DataFrame:
        """
            1. load dataset
            2. simplify column names
        :return: Dataset
        """
        df = self.load_dataset()
        columns = df.columns
        return pd.concat(map(lambda col: df[col].rename(col.split(" ")[0]), columns), axis=1)

    @property
    def columns(self) -> list:
        # header of dataset
        return self.dataset.columns.to_list()

    def process(self):
        """
            1. extract features
            2. train model and estimate metric
        :return: exit code
        """
        try:
            if self.p_type == "processed":
                processed = FeatureExtractor(
                    dataset=self.build_dataset()
                ).transform()
            else:
                processed = self.build_dataset()

            metric = self.model_pipeline(dataset=processed)
        except ImportError as e:
            # TODO: handling ImportError
            self.logger.exception(e)
            return False
        except Exception as e:
            self.logger.exception(e)
            return False

        return True

    def model_pipeline(self, dataset):
        """
            1. build dataset
            2. train model ( + gridsearch)
            3. estimate metrics
        :return: metric
        """
        # build dataset
        x_train, y_train, x_test, y_test = dataset

        # train model
        searcher = ElasticNetSearcher(
            x_train=x_train, y_train=y_train,
            columns = self.columns,
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
        searcher.save(prefix="result/{t}/".format(t=self.p_type))
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
