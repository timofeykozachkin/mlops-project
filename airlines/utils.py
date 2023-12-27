import subprocess


def rewrite_predictions(predictions_path="data/airlines_predictions.csv"):
    subprocess.run(["dvc", "remove", f"{predictions_path}.dvc"])
    subprocess.run(["dvc", "add", predictions_path])
    subprocess.run(["dvc", "push", predictions_path])
