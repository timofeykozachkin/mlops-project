import dvc.api
from sklearn.metrics import accuracy_score
import subprocess

from airlines.model import AirlinesCatBoost
from airlines.dataset import AirlinesDataset


subprocess.run(["dvc", "pull"])

dataset = AirlinesDataset()
model = AirlinesCatBoost(dataset.get_numeric_features(), 
                         dataset.get_categorical_features())

model.fit(dataset.get_train_features(), dataset.get_train_target())
