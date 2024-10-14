import joblib
import ast

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

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

    # Load the best hyperparameters from the hypertuning_results.txt file
    try:
        with open(args.hypertune_results_path, "r") as file:
            lines = file.readlines()
            params_line = next(
                line for line in lines if line.startswith("Best Hyperparameters:")
            )
            params_str = params_line.split(": ", 1)[1].strip()
            params = ast.literal_eval(params_str)  # Convert string to dictionary
    except Exception as e:
        print(f"Error reading the hyperparameters: {e}")
        params = {"n_estimators": 100}

    # Train the RandomForest model with the best hyperparameters on the full dataset
    model = RandomForestRegressor(**params, random_state=42, criterion="squared_error")
    model.fit(X_imputed, y)

    # Save the model, imputer for later inference
    joblib.dump(model, args.model_path)
    joblib.dump(imputer, args.imputer_path)

    # Model evaluation on test data
    y_pred = model.predict(X_imputed)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"Test RMSE: {rmse}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
