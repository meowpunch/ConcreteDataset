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
        self.columns = columns

        super().__init__(x_train, y_train, grid_params, score,
                         GradientBoostingRegressor(
                             learning_rate=0.15, min_samples_leaf=6, random_state=9))
