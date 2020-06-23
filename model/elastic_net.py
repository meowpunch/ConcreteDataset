import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from model.parent import BaseModel, BaseSearcher


class ElasticNetModel(BaseModel):
    """
        tune ElasticNet
    """

    def __init__(self, x_train, y_train, params=None):
        super().__init__(x_train, y_train, params, ElasticNet)

    @property
    def coef_df(self):
        """
        :return: pd DataFrame
        """
        return pd.Series(
            data=np.append(self.model.coef_, self.model.intercept_),
            index=self.x_train.columns.tolist() + ["intercept"],
        ).rename("beta").reset_index().rename(columns={"index": "column"})

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_coef(key="{prefix}/beta.pkl".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(prefix=prefix)
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_coef(self, key):
        self.logger.info("beta_coef:\n{coef}".format(coef=self.coef_df))
        self.coef_df.to_csv("coef".format(key))


class ElasticNetSearcher(BaseSearcher):
    """
        for gridsearch
    """

    def __init__(self, x_train, y_train, grid_params=None, score=mean_absolute_error):
        if grid_params is None:
            grid_params = {
                "max_iter": [1, 5, 10],
                "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)
            }

        self.x_train = x_train
        self.y_train = y_train
        self.scorer = score

        self.error = None  # pd.Series
        self.metric = None

        super().__init__(x_train, y_train, grid_params, score, ElasticNet)

    def fit(self, X=None, y=None, groups=None, **fit_params):
        super().fit(X=self.x_train, y=self.y_train)

    @property
    def coef_df(self):
        """
        :return: pd DataFrame
        """
        return pd.Series(
            data=np.append(self.best_estimator_.coef_, self.best_estimator_.intercept_),
            index=self.x_train.columns.tolist() + ["intercept"],
        ).rename("beta").reset_index().rename(columns={"index": "column"})

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_params(key="{prefix}/params.pkl".format(prefix=prefix))
        self.save_coef(key="{prefix}/beta.csv".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(prefix=prefix)
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_params(self, key):
        self.logger.info("tuned params: {params}".format(params=self.best_params_))

    def save_coef(self, key):
        self.logger.info("beta_coef:\n{coef}".format(coef=self.coef_df))
        self.coef_df.to_csv(key)
