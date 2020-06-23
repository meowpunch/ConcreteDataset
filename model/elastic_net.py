import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from model.parent import BaseSearcher


class ElasticNetSearcher(BaseSearcher):
    def __init__(self, x_train, y_train, columns, grid_params=None, score=mean_absolute_error):
        if grid_params is None:
            grid_params = {
                "max_iter": [10],
                "alpha": [10],
                "l1_ratio": [0]
            }
        self.columns = columns

        super().__init__(x_train, y_train, grid_params, score, ElasticNet())

    @property
    def coef_df(self):
        """
        :return: pd DataFrame
        """
        return pd.Series(
            data=np.append(self.best_estimator_.coef_, self.best_estimator_.intercept_),
            index=self.columns[:-1] + ["intercept"],
            name="beta"
        ).reset_index().rename(columns={"index": "column"})

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_params(key="{prefix}/params.pkl".format(prefix=prefix))
        self.save_coef(key="{prefix}/beta.csv".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(key="{prefix}/images/error_distribution.png".format(prefix=prefix))
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_coef(self, key):
        self.logger.info("beta_coef:\n{coef}".format(coef=self.coef_df))
        self.coef_df.to_csv(key, index=False)
