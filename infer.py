import subprocess

from airlines.dataset import AirlinesDataset
from airlines.model import AirlinesCatBoost


def infer():
    dataset = AirlinesDataset()
    model = AirlinesCatBoost(
        dataset.get_numeric_features(), dataset.get_categorical_features()
    )

    model.set_preprocessor("data/airlines_preprocessor.joblib")
    model.set_model("data/airlines_model")


if __name__ == "__main__":
    subprocess.run(["dvc", "pull"])
    infer()
