import ast
import joblib
from typing import Tuple, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils import root_mean_squared_error, parse_args


def load_data(train_data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and separate features and target from the dataset."""
    dataset_df = pd.read_csv(train_data_path)
    X = dataset_df.drop(columns=["target"])
    y = dataset_df["target"]
    return X, y


def handle_missing_values(X: pd.DataFrame) -> Tuple[pd.DataFrame, SimpleImputer]:
    """Impute missing values in the dataset."""
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    return X_imputed, imputer


def load_best_hyperparameters(hypertune_results_path: str) -> Dict:
    """Load best hyperparameters from hypertuning results file."""
    try:
        with open(hypertune_results_path, "r") as file:
            lines = file.readlines()
            params_line = next(
                line for line in lines if line.startswith("Best Hyperparameters:")
            )
            params_str = params_line.split(": ", 1)[1].strip()
            params = ast.literal_eval(params_str)  # Convert string to dictionary
    except Exception as e:
        print(f"Error reading the hyperparameters: {e}")
        params = {"n_estimators": 100}  # Fallback to default if there's an error
    return params


def train_model(
    X_imputed: pd.DataFrame, y: pd.Series, params: Dict
) -> RandomForestRegressor:
    """Train the model using the best hyperparameters."""
    model = RandomForestRegressor(**params, random_state=42, criterion="squared_error")
    model.fit(X_imputed, y)
    return model


def evaluate_model(
    model: RandomForestRegressor, X_imputed: pd.DataFrame, y: pd.Series
) -> None:
    """Evaluate the model using Root Mean Squared Error (RMSE)."""
    y_pred = model.predict(X_imputed)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"Test RMSE: {rmse}")


def save_model_and_imputer(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    model_path: str,
    imputer_path: str,
) -> None:
    """Save the trained model and imputer to disk."""
    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)


def main(args) -> None:
    """Main function to load data, train model, and save outputs."""
    # Load dataset and handle missing values
    X, y = load_data(args.train_data_path)
    X_imputed, imputer = handle_missing_values(X)

    # Load best hyperparameters
    params = load_best_hyperparameters(args.hypertune_results_path)

    # Train model on the full dataset
    model = train_model(X_imputed, y, params)

    # Evaluate the model
    evaluate_model(model, X_imputed, y)

    # Save model and imputer
    save_model_and_imputer(model, imputer, args.model_path, args.imputer_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
