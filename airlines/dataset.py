import pandas as pd
from sklearn.model_selection import train_test_split


class AirlinesDataset:
    def __init__(self):
        self.data = pd.read_csv("data/airlines_data.csv").set_index("id")
        self.numeric_features = list(
            set(self.data.select_dtypes(include="number").columns)
        )
        self.categorical_features = list(
            self.data.select_dtypes(include="object").columns[
                self.data.select_dtypes(include="object").columns != "satisfaction"
            ]
        )
        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            self.X_test,
        ) = self.train_test_split()

    def train_test_split(self, target_col="satisfaction", test_mark="-"):
        df = self.get_data()
        df_train = df.loc[df[target_col] != test_mark, :]
        X_test = df.loc[df[target_col] == test_mark, df.columns != target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            df_train.loc[:, df_train.columns != target_col],
            df_train[target_col],
            test_size=0.15,
            random_state=0,
        )
        return X_train, X_val, y_train, y_val, X_test

    def get_data(self):
        return self.data

    def get_train_features(self):
        return self.X_train

    def get_train_target(self):
        return self.y_train

    def get_val_features(self):
        return self.X_val

    def get_val_target(self):
        return self.y_val

    def get_test_features(self):
        return self.X_test

    def get_numeric_features(self):
        return self.numeric_features

    def get_categorical_features(self):
        return self.categorical_features
