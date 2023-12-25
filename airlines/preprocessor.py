from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class AirlinesPreprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def get_numeric_features(self):
        return self.numeric_features

    def get_categorical_features(self):
        return self.categorical_features

    def call_numeric_transformer(self):
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), 
                ("scaler", StandardScaler())
                ]
        )
        return numeric_transformer

    def call_categorical_transformer(self):
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        return categorical_transformer

    def __call__(self):
        numeric_transformer = self.call_numeric_transformer()
        categorical_transformer = self.call_categorical_transformer()
        numeric_features = self.get_numeric_features()
        categorical_features = self.get_categorical_features()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        return preprocessor
