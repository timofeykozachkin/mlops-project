import pandas as pd

class AirlinesDataset:
    def __init__(self):
        self.data = pd.read_csv('data/airlines_data.csv').set_index('id')
        self.numeric_features = list(set(self.data.select_dtypes(include='number').columns))
        self.categorical_features = list(
            self.data.select_dtypes(include='object')\
            .columns[self.data.select_dtypes(include='object').columns != 'satisfaction'])

    def get_data(self):
        return self.data

    def get_train_features(self, target_col = 'satisfaction', test_mark = '-'):
        df = self.get_data()
        X_train = df.loc[df[target_col] != test_mark, df.columns != target_col]
        return X_train

    def get_train_target(self, target_col = 'satisfaction', test_mark = '-'):
        df = self.get_data()
        y_train = df.loc[df[target_col] != test_mark, target_col]
        return y_train

    def get_test_features(self, target_col = 'satisfaction', test_mark = '-'):
        df = self.get_data()
        X_test = df.loc[df[target_col] == test_mark, df.columns != target_col]
        return X_test

    def get_numeric_features(self):
        return self.numeric_features

    def get_categorical_features(self):
        return self.categorical_features
