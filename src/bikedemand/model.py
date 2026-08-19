from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

"""def create_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators = 300,
        min_samples_leaf = 5,
        max_features = "sqrt",
        random_state = 42,
        n_jobs = -1,
    )
"""
def create_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )

def train_model(model, x, y):
    model.fit(x, y)
    return model
