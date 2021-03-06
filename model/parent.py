import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

from utils.logger import init_logger
from utils.visualize import draw_hist


class BaseSearcher(GridSearchCV):
    """
        BaseSearcher for ElasticNetSearcher & GradientBoostSearcher
    """

    def __init__(
            self, x_train, y_train,
            grid_params=None, score=mean_squared_error,
            estimator=ElasticNet()
    ):
        if grid_params is None:
            raise ValueError("grid params are needed")

        self.x_train = x_train
        self.y_train = y_train
        self.scorer = score

        self.error = None  # pd.Series
        self.metric = None

        # logger
        self.logger = init_logger()

        super().__init__(
            estimator=estimator,
            param_grid=grid_params,
            scoring=make_scorer(self.scorer, greater_is_better=False)
        )

    def fit(self, X=None, y=None, groups=None, **fit_params):
        super().fit(X=self.x_train, y=self.y_train)

    def estimate_metric(self, y_true, y_pred, prefix):
        self.error = pd.Series(y_true - y_pred).rename("error")
        true = pd.Series(y_true)
        self.save_error_ratio(err_ratio=(abs(self.error) / true) * 100, true=true,
                              key="{p}/images/error_ratio.png".format(p=prefix))

        self.metric = self.scorer(y_true=y_true, y_pred=y_pred)
        return self.metric

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model

            # TODO: save to s3
        :param prefix: pr
        """
        self.save_params(key="{prefix}/best_params.pkl".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(key="{prefix}/images/error_distribution.png".format(prefix=prefix))
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_params(self, key):
        self.logger.info("tuned params: {params}".format(params=self.best_params_))
        dump(self.best_params_, key)

    def save_metric(self, key):
        self.logger.info("metric is {metric}".format(metric=self.metric))

    def save_model(self, key):
        # save best elastic net
        dump(self.best_estimator_, key)

    def save_error_distribution(self, key):
        draw_hist(self.error)
        plt.savefig(key)

    def save_error_ratio(self, err_ratio, true, key):
        plt.figure()

        plt.scatter(x=true, y=err_ratio)
        plt.xlabel("y_true")
        plt.ylabel("err_ratio = abs(y_true - y_pred) / y_true")
        plt.title("err_ratio by y_true")
        plt.savefig(key)
        self.logger.info("mean of err_ratio: {v}".format(v=err_ratio.mean()))
