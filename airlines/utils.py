import os
import socket
import subprocess

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve


def set_mlflow_tracking():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("128.0.1.1", 8080))
    if result == 0:
        mlflow.set_tracking_uri(uri="http://128.0.1.1:8080")
    else:
        print("MLFlow execution is on uri=127.0.0.1:8080")
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    sock.close()


def rewrite_predictions(predictions_path="data/airlines_predictions.csv"):
    if os.path.exists(f"{predictions_path}.dvc"):
        subprocess.run(["dvc", "remove", f"{predictions_path}.dvc"])
    subprocess.run(["dvc", "add", predictions_path])
    subprocess.run(["dvc", "push", predictions_path])


def roc_plot(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1], pos_label="satisfied")

    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("data/pics/ROC-Curve.png")


def loss_plot(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Log-loss")
    plt.title("Loss")
    plt.savefig("data/pics/Log-loss.png")


def cm_plot(y_true, predictions, labels):
    cm = confusion_matrix(y_true, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.tight_layout()
    plt.savefig("data/pics/Confusion-Matrix.png")
