from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from model.parent import BaseSearcher


class GradientBoostSearcher(BaseSearcher):
    def __init__(self, x_train, y_train, columns, grid_params=None, score=mean_absolute_error):
        if grid_params is None:
            grid_params = {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [6],
                'min_samples_leaf': [3],
                'max_features': [1.0]
            }

        self.x_train = x_train
        self.y_train = y_train
        self.columns = columns
        self.scorer = score

        self.error = None  # pd.Series
        self.metric = None

        super().__init__(x_train, y_train, grid_params, score,
                         GradientBoostingRegressor(
                             learning_rate=0.15, min_samples_leaf=6, random_state=9))

    def fit(self, X=None, y=None, groups=None, **fit_params):
        super().fit(X=self.x_train, y=self.y_train)

    def save(self, prefix):
        """
            save tuned params, beta coef, metric, distribution, model
        :param prefix: dir
        """
        self.save_params(key="{prefix}/params.pkl".format(prefix=prefix))
        self.save_metric(key="{prefix}/metric.pkl".format(prefix=prefix))
        self.save_error_distribution(key="{prefix}/images/error_distribution.png".format(prefix=prefix))
        self.save_model(key="{prefix}/model.pkl".format(prefix=prefix))

    def save_params(self, key):
        self.logger.info("tuned params: {params}".format(params=self.best_params_))
