import subprocess

from airlines.dataset import AirlinesDataset
from airlines.model import AirlinesCatBoost

subprocess.run(["dvc", "pull"])

dataset = AirlinesDataset()
model = AirlinesCatBoost(
    dataset.get_numeric_features(), dataset.get_categorical_features()
)

model.fit(dataset.get_train_features(), dataset.get_train_target())
