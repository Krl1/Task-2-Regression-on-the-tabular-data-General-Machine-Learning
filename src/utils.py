import argparse
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the root mean squared error (RMSE).

    Parameters:
    - y_true: array-like of shape (n_samples,) - Ground truth target values.
    - y_pred: array-like of shape (n_samples,) - Predicted target values.

    Returns:
    - RMSE: float - Root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
    - args: Namespace - Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Train a RandomForest model")
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the train dataset",
        default="data/train.csv",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to the test dataset",
        default="data/hidden_test.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model",
        default="results/random_forest_model.pkl",
    )
    parser.add_argument(
        "--imputer_path",
        type=str,
        help="Path to the Imputer model",
        default="results/imputer_model.pkl",
    )
    parser.add_argument(
        "--hypertune_results_path",
        type=str,
        help="Path to the hypertuning_results.txt file",
        default="results/hypertuning_results.txt",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of iterations for RandomizedSearchCV",
        default=10,
    )
    parser.add_argument(
        "--prediction_output_path",
        type=str,
        help="Path to save the predictions",
        default="results/predictions.csv",
    )

    return parser.parse_args()
