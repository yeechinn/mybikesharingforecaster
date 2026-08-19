import pandas as pd 
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass

from bikedemand.features import TARGET

@dataclass(frozen=True) 
class DataSplit: 
    """Chronological train, validation and test datasets.""" 
    train: pd.DataFrame 
    test: pd.DataFrame
    val: pd.DataFrame

def chronological_split( 
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    ) -> DataSplit: 
    """Split data chronologically. 
    No observations from the future are included in earlier splits. """
    #train_end = pd.Timestamp(train_end)
    if not df["dteday"].is_monotonic_increasing: raise ValueError("Data must be sorted by datetime.") 
    if train_end > val_end: 
        raise ValueError("validation end date must be after training end date.")
    train = df[df["dteday"] < train_end].copy() 
    val = df[(df["dteday"] >= train_end) &  (df["dteday"] < val_end)].copy()
    test = df[df["dteday"] >= val_end].copy()
    return DataSplit( train=train, val=val, test=test)

def calculate_mae( 
        y_true: pd.Series, 
        y_pred: pd.Series, 
        ) -> float: 
     """Calculate mean absolute error.""" 
     if len(y_true) != len(y_pred): 
        raise ValueError("y_true and y_pred must have the same length.") 
     return mean_absolute_error(y_true, y_pred)

def evaluate_model( 
        df: pd.DataFrame, 
        ) -> float: 
    return calculate_mae(df[TARGET], df['predicted_demand'])
