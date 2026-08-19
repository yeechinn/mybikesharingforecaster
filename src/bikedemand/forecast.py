from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Sequence
import joblib

from bikedemand.data import load_data_training, load_data_forecast
from bikedemand.features import create_features, FEATURES, TARGET
from bikedemand.model import create_model, train_model
from bikedemand.evaluation import calculate_mae, evaluate_model, chronological_split

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict hourly bike-sharing demand from CSV input.")

    parser.add_argument(
        "--train-data",
        type=Path,
        help="CSV file containing historical bike-sharing observations used to train the model.",
    )

    parser.add_argument( 
        "--model", 
        type=Path, 
        help="Path for the model.", 
    )

    parser.add_argument( 
        "--input", 
        type=Path, 
        help="Path for the input CSV.", 
    )

    parser.add_argument( 
        "--output", 
        default=Path(__file__).resolve().parents[2] / "data" / "hour_prediction.csv",
        type=Path, 
        help="Path for the prediction CSV.", 
    )

    return parser

def train_and_evaluate(df: pd.DataFrame) -> None:

    featured_data = create_features(df)

    split = chronological_split(
        featured_data, 
        train_end="2012-09-01", 
        val_end="2012-11-01",
    )
    if len(split.train.index) == 0 or len(split.val.index) == 0 or len(split.test.index) == 0:
        raise ValueError("Empty train, val or test data. Check input training data or chronological split dates")

    X_train, y_train = split.train[FEATURES], split.train[TARGET]
    X_val = split.val[FEATURES]
    X_test = split.test[FEATURES]

    print("Training a forecast model.")
    model = create_model()
    model = train_model(model, X_train, y_train)

    # Training performance
    predictions_train = model.predict(X_train)
    train_mae = calculate_mae(split.train[TARGET], predictions_train)

    # Validation performance
    predictions_val = model.predict(X_val)
    val_mae = calculate_mae(split.val[TARGET], predictions_val)

    # Final test performance
    predictions = predict(model, X_test) 
    results = split.test.copy()
    results["predicted_demand"] = predictions
    test_mae = evaluate_model(results)

    # As baseline, use demand from previous week
    baseline_mae = calculate_mae(split.test[TARGET], split.test["demand_previousweek"])

    print(f"Training MAE:   {train_mae:.2f}")
    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Test MAE:       {test_mae:.2f}")
    print(f"Baseline MAE:   {baseline_mae:.2f}")

    print("\nFeature importance:")
    for feature, importance in zip(
        FEATURES,
        model.feature_importances_,
    ):
        print(f"{feature}: {importance:.3f}")

    joblib.dump(model, "model.pkl") # save the trained model
    split.test.to_csv(Path(__file__).resolve().parents[2] / "data" / "hour_test.csv") # save the test dataset for future evaluation


def load_model(model_path: Path):
    """Load a trained model from disk.""" 
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}") 
    return joblib.load(model_path)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Generate demand predictions."""
    predictions = model.predict(df)
    predictions = np.clip(np.asarray(predictions, dtype=float), a_min=0.0, a_max=None)  # rental count cannot be negative
    return predictions


def save_predictions( results: pd.DataFrame, output_path: Path, ) -> None: 
    """Save predictions to CSV.""" 
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    results.to_csv(output_path, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
   
    if (args.train_data):
        """ Load data, split into train, val and test dataset, create the input features and train on the training dataset.
        Evaluate performance on val and test dataset """ 

        data = load_data_training(args.train_data)
        print("Data loaded successfully.")
        train_and_evaluate(data)

    elif (args.model and args.input):
        """ Use the loaded model to predict input csv. """

        print("Loading model and input csv.")
        model = load_model(args.model)
        data = load_data_forecast(args.input) 
        X_test = data[FEATURES]

        print("Running the forecast.")
        predictions = predict( model, X_test ) 
        results = data.copy()
        results["predicted_demand"] = predictions

        print("Saving to file.")
        save_predictions(results=results, output_path=args.output)

    else:
        raise ValueError(f"Please provide either the train dataset via --train-data or the model and the input csv via --model and --input.")



if __name__ == "__main__":
    main()

