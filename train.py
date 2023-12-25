import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

from airlines.dataset import AirlinesDataset
from airlines.model import AirlinesCatBoost


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    dataset = AirlinesDataset()
    params = OmegaConf.to_container(cfg["params"])
    model = AirlinesCatBoost(
        dataset.get_numeric_features(), dataset.get_categorical_features()
    )
    model.set_model_params(params)

    model.fit(dataset.get_train_features(), dataset.get_train_target())


if __name__ == "__main__":
    subprocess.run(["dvc", "pull"])
    train()
