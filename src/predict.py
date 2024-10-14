import joblib

import pandas as pd

from utils import parse_args


def main(args):
    # Load the saved model and imputer
    model = joblib.load(args.model_path)
    imputer = joblib.load(args.imputer_path)

    # Load test data
    test_df = pd.read_csv(args.test_data_path)

    # Check if target column is present in the test data
    if "target" in test_df.columns:
        test_df.drop(columns=["target"], inplace=True)
        print("Target column dropped from the test data")

    # Preprocessing: Handle missing values
    X_test_imputed = imputer.transform(test_df)

    # Make predictions
    predictions = model.predict(X_test_imputed)

    # Display predictions
    print("Predictions:")
    print(predictions)

    # Optionally save predictions to a file
    output_path = args.prediction_output_path
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
