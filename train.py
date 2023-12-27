import subprocess

import hydra
import joblib
import mlflow
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score

from airlines.dataset import AirlinesDataset
from airlines.model import AirlinesCatBoost
from airlines.utils import cm_plot, loss_plot, roc_plot, set_mlflow_tracking

set_mlflow_tracking()
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("/airlines")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    with mlflow.start_run(run_name="airlines_catboost"):
        # 0. set tag for mlflow
        mlflow.set_tag("model_name", "airlines_catboost")

        # 1. model setting
        dataset = AirlinesDataset()
        params = OmegaConf.to_container(cfg["params"])
        model = AirlinesCatBoost(
            dataset.get_numeric_features(), dataset.get_categorical_features()
        )

        # 2. hyperparameters setting and model fitting
        model.set_model_params(params)
        model.fit(dataset.get_train_features(), dataset.get_train_target())

        # 3. save model
        model.get_model().save_model("model/airlines_model", format="cbm")
        joblib.dump(model.get_preprocessor_pipe(), "model/airlines_preprocessor.joblib")

        # 4.1. logging model
        mlflow.catboost.log_model(model.get_model(), "cb_models")
        mlflow.log_params(params=params)
        # mlflow.log_artifact('catboost_info/catboost_training.json')

        # 4.2. logging metrics, loss func
        predictions = model.predict(dataset.get_val_features())
        y_scores = model.predict_proba(dataset.get_val_features())

        acc = accuracy_score(dataset.get_val_target(), predictions)
        roc_auc = roc_auc_score(dataset.get_val_target(), y_scores[:, 1])
        roc_plot(dataset.get_val_target(), y_scores)
        loss_plot(model.get_model().get_evals_result()["learn"]["Logloss"])
        cm_plot(dataset.get_val_target(), predictions, model.get_model().classes_)

        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("ROC-AUC score", roc_auc)
        mlflow.log_artifact("data/pics/ROC-Curve.png")
        mlflow.log_artifact("data/pics/Log-loss.png")
        mlflow.log_artifact("data/pics/Confusion-Matrix.png")


if __name__ == "__main__":
    subprocess.run(["dvc", "pull", "data/airlines_data.csv.dvc"])
    train()
