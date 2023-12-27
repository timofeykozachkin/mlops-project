import subprocess

import pandas as pd

from airlines.dataset import AirlinesDataset
from airlines.model import AirlinesCatBoost
from airlines.utils import rewrite_predictions


def infer():
    dataset = AirlinesDataset()
    model = AirlinesCatBoost(
        dataset.get_numeric_features(), dataset.get_categorical_features()
    )

    model.set_preprocessor("model/airlines_preprocessor.joblib")
    model.set_model("model/airlines_model")

    preds = model.predict(dataset.get_test_features())
    pd.Series(preds, name="predictions").to_csv(
        "data/airlines_predictions.csv", index=False
    )


if __name__ == "__main__":
    subprocess.run(["dvc", "pull"])
    infer()
    rewrite_predictions()
