class FeatureExtractor:
    def __init__(self, dataset):
        self.x_train = x_train
        self.x_test = x_test

        numeric_features = ['Age', 'Fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['Embarked', 'Sex', 'Pclass']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.column_transformer = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    def process(self):
        return True


    def preprocess_x(self):
        pass

    def preprocess_y(self):
        pass