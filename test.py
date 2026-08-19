import unittest
import pandas as pd
from pathlib import Path

from src.bikedemand.features import create_features
from src.bikedemand.evaluation import chronological_split

DATA_PATH = Path(__file__).resolve().parents[0] / "data" / "hour.csv"

""" A test data """
df = pd.DataFrame({
    "dteday": ["2011-01-01", "2011-01-05", "2012-01-06", "2012-01-18"],
    "hr": [8, 8, 17, 17],
    "atemp": [0.1, 0.2, 0.3, 0.4],
    "hum": [0.1, 0.2, 0.3, 0.4],
    "weathersit": [1, 2, 3, 4],
    "weekday": [0, 1, 2, 3],
    "workingday": [0, 1, 0, 1],
    "holiday": [0, 0, 1, 0],
    "cnt": [20, 20, 20, 20],
    "demand_previousday": [10, 10, 10, 10],
    "demand_previousweek": [10, 10, 10, 10],
})


class Test(unittest.TestCase):

    def test_hour_workingday_feature(self):
        featured_data = create_features(df)
        self.assertEqual( featured_data["hourxworkingday"].tolist(), [ 8, 32, 17, 41,], "hour x working day feature not as expected.")

    def test_chronological_split_has_no_overlap(self):
        split = chronological_split(
            df, 
            train_end="2012-01-05",
            val_end="2012-01-07",
        )
        self.assertTrue( split.train["dteday"].max() < split.val["dteday"].min(), "train and val data overlapping")
        self.assertTrue( split.val["dteday"].max() < split.test["dteday"].min(), "val and test data overlapping")

if __name__ == "__main__":
    unittest.main()
