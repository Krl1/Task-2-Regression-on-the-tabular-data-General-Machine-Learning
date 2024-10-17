import joblib
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from scipy.stats import randint

from utils import root_mean_squared_error, parse_args


def load_and_preprocess_data(
    train_data_path: str,
) -> Tuple[pd.DataFrame, pd.Series, SimpleImputer]:
    """Load and preprocess the dataset."""
    dataset_df = pd.read_csv(train_data_path)

    X = dataset_df.drop(columns=["target"])
    y = dataset_df["target"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, y, imputer


def perform_hyperparameter_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, n_iter: int
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """Perform RandomizedSearchCV to find the best model parameters."""
    rf_model = RandomForestRegressor(criterion="squared_error", random_state=42)

    param_dist = {
        "n_estimators": randint(10, 200),
        "max_depth": randint(5, 50),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }

    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


def save_model_and_results(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    model_path: str,
    imputer_path: str,
    results_path: str,
    best_params: Dict[str, Any],
    rmse: float,
) -> None:
    """Save the trained model, imputer, and tuning results to disk."""
    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)

    with open(results_path, "w") as f:
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Test RMSE: {rmse}\n")


def main(args: Any) -> None:
    """Main function to run the training and evaluation process."""
    try:
        # Load and preprocess data
        X_imputed, y, imputer = load_and_preprocess_data(args.train_data_path)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )

        # Perform hyperparameter tuning
        best_model, best_params = perform_hyperparameter_tuning(
            X_train, y_train, args.n_iter
        )

        print("Best Hyperparameters:", best_params)

        # Evaluate the model
        rmse = evaluate_model(best_model, X_test, y_test)
        print(f"Test RMSE: {rmse}")

        # Save model, imputer, and tuning results
        save_model_and_results(
            best_model,
            imputer,
            args.model_path,
            args.imputer_path,
            args.hypertune_results_path,
            best_params,
            rmse,
        )

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
