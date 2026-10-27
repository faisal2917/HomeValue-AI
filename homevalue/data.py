"""Loading and splitting the California Housing dataset."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "MedHouseVal"

# Column order matters: the model is trained on (and must predict from) exactly this order.
FEATURES = [
    "MedInc",      # median income in the block group, in units of $10,000
    "HouseAge",    # median house age in years
    "AveRooms",    # average rooms per household
    "AveBedrms",   # average bedrooms per household
    "Population",  # people living in the block group
    "AveOccup",    # average people per household
    "Latitude",
    "Longitude",
]

# The target is the median house value in units of $100,000, capped at 5.00001 ($500,001).
TARGET_UNIT_DOLLARS = 100_000

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Return the full dataset as one DataFrame: 8 feature columns plus the target."""
    return fetch_california_housing(as_frame=True).frame


def split_features_target(df: pd.DataFrame):
    """Separate the feature matrix X from the target vector y."""
    return df[FEATURES], df[TARGET]


def train_test_data(df: pd.DataFrame):
    """Hold out 20% of rows as unseen test data. A fixed seed keeps the split reproducible."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
