import joblib
from typing import Tuple, Any

import pandas as pd

from utils import parse_args


def load_model_and_imputer(model_path: str, imputer_path: str) -> Tuple[Any, Any]:
    """Load the trained model and imputer from disk."""
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    return model, imputer


def preprocess_data(test_df: pd.DataFrame, imputer: Any) -> pd.DataFrame:
    """Preprocess the test data by handling missing values."""
    # Drop target column if present
    if "target" in test_df.columns:
        test_df = test_df.drop(columns=["target"])
        print("Target column dropped from the test data")

    # Impute missing values
    X_test_imputed = imputer.transform(test_df)
    return X_test_imputed


def make_predictions(model: Any, X_test_imputed: pd.DataFrame) -> pd.Series:
    """Make predictions using the trained model."""
    predictions = model.predict(X_test_imputed)
    return predictions


def save_predictions(predictions: pd.Series, output_path: str) -> None:
    """Save predictions to a CSV file."""
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main(args: Any) -> None:
    """Main function to load data, make predictions, and save results."""
    try:
        # Load model and imputer
        model, imputer = load_model_and_imputer(args.model_path, args.imputer_path)

        # Load and preprocess test data
        test_df = pd.read_csv(args.test_data_path)
        X_test_imputed = preprocess_data(test_df, imputer)

        # Make predictions
        predictions = make_predictions(model, X_test_imputed)

        # Save predictions
        save_predictions(predictions, args.prediction_output_path)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
