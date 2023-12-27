import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

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
                ("classifier", CatBoostClassifier()),
            ]
        )
        return clf

    def set_model_params(self, params):
        self.model["classifier"].set_params(**params)

    def set_model(self, model_path):
        self.model["classifier"].load_model(model_path)

    def get_model(self):
        return self.model["classifier"]

    def get_preprocessor(self):
        return self.preprocessor

    def get_preprocessor_pipe(self):
        return self.model["preprocessor"]

    def set_preprocessor(self, preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        self.model.steps.pop(0)
        self.model.steps.insert(0, ["preprocessor", preprocessor])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_val):
        return self.model.predict(X_val)

    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)
