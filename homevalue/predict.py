"""Loading the saved model, validating user input, and producing a price estimate."""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from homevalue.data import FEATURES, MODEL_PATH, TARGET_UNIT_DOLLARS

# The dataset caps values at $500,001, so the model cannot predict meaningfully above it.
PRICE_CAP_DOLLARS = 500_001

# Median MedHouseVal across all 20,640 block groups, used as a statewide reference point.
STATE_MEDIAN_DOLLARS = 179_700

# Rough bounding box of California; coordinates outside it are rejected.
CA_LAT_RANGE = (32.5, 42.0)
CA_LON_RANGE = (-124.5, -114.1)

FEATURE_LABELS = {
    "MedInc": "Median household income",
    "HouseAge": "Median house age",
    "AveRooms": "Average rooms per home",
    "AveBedrms": "Average bedrooms per home",
    "Population": "Neighborhood population",
    "AveOccup": "Average occupants per home",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Load the dict saved by scripts/train.py (pipeline plus metadata)."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python scripts/train.py` first."
        )
    return joblib.load(path)


def validate_inputs(values: Dict[str, float],
                    feature_ranges: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors block a prediction; warnings only inform the user."""
    errors = []
    warnings = []

    missing = [f for f in FEATURES if values.get(f) is None]
    if missing:
        return [f"Missing value for: {', '.join(FEATURE_LABELS[f] for f in missing)}."], []

    for f in ("MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"):
        if values[f] <= 0:
            errors.append(f"{FEATURE_LABELS[f]} must be greater than zero.")
    if values["HouseAge"] < 0:
        errors.append("Median house age cannot be negative.")
    if values["AveBedrms"] > values["AveRooms"]:
        errors.append("Average bedrooms cannot exceed average rooms.")
    if not CA_LAT_RANGE[0] <= values["Latitude"] <= CA_LAT_RANGE[1]:
        errors.append(f"Latitude must be within California ({CA_LAT_RANGE[0]} to {CA_LAT_RANGE[1]}).")
    if not CA_LON_RANGE[0] <= values["Longitude"] <= CA_LON_RANGE[1]:
        errors.append(f"Longitude must be within California ({CA_LON_RANGE[0]} to {CA_LON_RANGE[1]}).")

    if not errors:
        for f in FEATURES:
            lo, hi = feature_ranges[f]["min"], feature_ranges[f]["max"]
            if not lo <= values[f] <= hi:
                warnings.append(
                    f"{FEATURE_LABELS[f]} is outside the range the model was trained on, "
                    "so this estimate is less reliable."
                )
    return errors, warnings


def predict_price(artifact: dict, values: Dict[str, float]) -> float:
    """Return the estimated median house value in dollars."""
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    prediction = float(artifact["pipeline"].predict(row)[0])
    return max(prediction, 0.0) * TARGET_UNIT_DOLLARS


def format_currency(dollars: float) -> str:
    return f"${dollars:,.0f}"
