from pathlib import Path
import pandas as pd

from bikedemand.features import create_features, FEATURES

def load_data_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {
        "dteday",
        "hr",
        "cnt",
        "workingday",
        "weathersit",
        "weekday",
        "holiday",
        "atemp",
        "hum",
    }

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

def load_data_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = FEATURES

    missing = [column for column in required_columns if column not in df.columns]

    if missing ==  ['demand_previousday', 'demand_previousweek', 'hourxworkingday']:
        df = create_features(df)

    elif missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df