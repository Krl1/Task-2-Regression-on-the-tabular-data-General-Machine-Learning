import joblib

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from scipy.stats import randint

from utils import root_mean_squared_error, parse_args


def main(args):
    # Load dataset
    dataset_df = pd.read_csv(args.train_data_path)

    # Separate features and target
    X = dataset_df.drop(columns=["target"])
    y = dataset_df["target"]

    # Preprocessing: Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # Define the RandomForest model
    rf_model = RandomForestRegressor(criterion="squared_error", random_state=42)

    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        "n_estimators": randint(10, 200),  # Number of trees in the forest
        "max_depth": randint(5, 50),  # Maximum depth of the tree
        "min_samples_split": randint(
            2, 20
        ),  # Minimum number of samples required to split an internal node
        "min_samples_leaf": randint(
            1, 10
        ),  # Minimum number of samples required to be at a leaf node
        "max_features": [
            "sqrt",
            "log2",
        ],  # Number of features to consider for best split
        "bootstrap": [
            True,
            False,
        ],  # Whether bootstrap samples are used when building trees
    }

    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=args.n_iter,  # Number of different hyperparameter combinations to try
        scoring=rmse_scorer,  # Metric to optimize (RMSE)
        cv=5,  # 5-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best model and hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the test data
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Test RMSE: {rmse}")

    # Save the best model and imputer
    joblib.dump(best_model, args.model_path)
    joblib.dump(imputer, args.imputer_path)

    # Optionally save the RMSE and best hyperparameters to a file
    with open(args.hypertune_results_path, "w") as f:
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Test RMSE: {rmse}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
