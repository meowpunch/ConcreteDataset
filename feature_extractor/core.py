from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer


class FeatureExtractor:
    def __init__(self, dataset):
        self.x_train, self.y_train, self.x_test, self.y_test = dataset

        self.numerical_features = ['Cement', 'Water', 'Coarse', 'Fine', 'Age']
        self.sparse_features = ['Blast', 'Fly', 'Superplasticizer']

        self.output_feature = 'Concrete'

        quantile_tf = QuantileTransformer(output_distribution='normal')
        sparse_tf = FunctionTransformer(func=lambda df: df.applymap(lambda x: 1 if x > 0 else 0))

        self.column_transformer = ColumnTransformer(transformers=[
            ('quantile', quantile_tf, self.numerical_features),
            ('sparse', sparse_tf, self.sparse_features)
        ])

    def fit(self):
        return self.column_transformer.fit(self.x_train)

    def transform(self):
        self.fit()
        train = self.column_transformer.transform(X=self.x_train)
        test = self.column_transformer.transform(X=self.x_test)
        return train, self.y_train, test, self.y_test
