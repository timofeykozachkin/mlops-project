from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from .preprocessor import AirlinesPreprocessor


class AirlinesCatBoost:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = AirlinesPreprocessor(numeric_features, categorical_features)
        self.model = self.init_model()

    def init_model(self):
        preprocessor = self.get_preprocessor()
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor()), 
                ("classifier", CatBoostClassifier())
            ]
        )
        return clf

    def get_model(self):
        return self.model

    def get_preprocessor(self):
        return self.preprocessor

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_val):
        return self.model.predict(X_val)
