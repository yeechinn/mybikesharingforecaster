import pandas as pd

TARGET = ["cnt",]
DATE = ["dteday",]
FEATURES = [
    "hr",
    "atemp",
    "hum",
    "weathersit",
    "weekday",
    "holiday",
    "demand_previousday",
    "demand_previousweek",
    "hourxworkingday",
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    """ Combine weathersit = 4 and weathersit = 3 as there are too few data points with weathersit = 4 """
    df["weathersit"] = df["weathersit"].replace({
        1: 1,  # clear
        2: 2,  # mist/cloudy
        3: 3,  # light precipitation
        4: 3,   # heavy precipitation -> combine
    })
            
    """ New features """
    """ Cannot use shift due to occasional missing hourly data, instead calculate from date. """
    #df["demand_previousday"] = df["cnt"].shift(24) # demand from 24 hour ago
    #df["demand_previousweek"] = df["cnt"].shift(168) # demand from exactly a week ago (this usually follows the same pattern except for holidays)

    if "demand_previousday" not in df.columns:
        df = add_lag_demand(df, 1, "demand_previousday")
    if "demand_previousweek" not in df.columns:
        df = add_lag_demand(df, 7, "demand_previousweek")
    if "hourxworkingday" not in df.columns:
        df["hourxworkingday"] = df["hr"] + 24 * df["workingday"] # 0-23 for non working day, 24-47 for working day

    df.dropna(subset=["demand_previousday", "demand_previousweek"], inplace=True) # delete rows for these features are empty

    return df

def add_lag_demand(df, days, column_name) -> pd.DataFrame:
    lagged = df[["dteday", "hr", "cnt"]].copy()

    lagged["dteday"] = (pd.to_datetime(lagged["dteday"]) + pd.Timedelta(days=days)).astype(str)

    lagged = lagged.rename(
        columns={"cnt": column_name}
    )

    return df.merge(
        lagged,
        on=["dteday", "hr"],
        how="left",
    )
