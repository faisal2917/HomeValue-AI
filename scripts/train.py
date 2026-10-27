"""Phases 2-4 (and the save step of Phase 6): prepare data, train three regressors,
evaluate them on held-out test data, and save the best one with joblib.

Run from the project root:  python scripts/train.py
Writes models/house_price_model.joblib and reports/metrics.json.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from homevalue.data import (
    FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_UNIT_DOLLARS,
    load_dataset,
    train_test_data,
)

# NumPy 2.0 with macOS's Accelerate BLAS emits spurious divide/overflow warnings from
# LinearRegression's matrix multiply; the results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model")


def build_models() -> dict:
    """Each model is a Pipeline so any preprocessing is saved together with the estimator.

    Scaling: LinearRegression's predictions do not change with feature scale, but scaling
    makes its coefficients comparable, so it gets a StandardScaler. Trees split on
    thresholds, so scaling has no effect on them and they use the raw features.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        # An unrestricted tree memorises the training data; a depth limit and a minimum
        # leaf size make it generalise better.
        "Decision Tree": Pipeline([
            ("model", DecisionTreeRegressor(max_depth=12, min_samples_leaf=10,
                                            random_state=RANDOM_STATE)),
        ]),
        # min_samples_leaf keeps the saved file small enough to deploy (fully grown trees
        # produce a model file of several hundred MB) at almost no cost in accuracy.
        "Random Forest": Pipeline([
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                            n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(y_true, y_pred) -> dict:
    """MAE and RMSE are in the target's units ($100k); they are also reported in dollars."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAE_dollars": round(float(mae) * TARGET_UNIT_DOLLARS),
        "RMSE_dollars": round(rmse * TARGET_UNIT_DOLLARS),
    }


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_data(df)
    print(f"Training rows: {len(X_train):,}   Test rows (unseen): {len(X_test):,}\n")

    results = {}
    fitted = {}
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results[name] = {
            "train": evaluate(y_train, pipeline.predict(X_train)),
            "test": evaluate(y_test, pipeline.predict(X_test)),
        }

    header = f"{'Model':<20}{'Train R2':>10}{'Test R2':>10}{'Test MAE':>12}{'Test RMSE':>12}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<20}{r['train']['R2']:>10.3f}{r['test']['R2']:>10.3f}"
              f"{'$' + format(r['test']['MAE_dollars'], ','):>12}"
              f"{'$' + format(r['test']['RMSE_dollars'], ','):>12}")
    print("\nA large gap between train and test R2 means the model memorised the training data.")

    # Lowest test RMSE wins: RMSE penalises large misses, which matter most for a price estimate.
    best_name = min(results, key=lambda n: results[n]["test"]["RMSE"])
    best = fitted[best_name]
    print(f"\nSelected model: {best_name} (lowest test RMSE)")

    # Refit on all data now that the model choice is made, so the deployed model sees every row.
    X_all, y_all = df[FEATURES], df["MedHouseVal"]
    best.fit(X_all, y_all)

    artifact = {
        "pipeline": best,
        "model_name": best_name,
        "features": FEATURES,
        "test_metrics": results[best_name]["test"],
        # Observed ranges let the app warn when an input is outside what the model has seen.
        "feature_ranges": {
            col: {"min": float(X_all[col].min()), "max": float(X_all[col].max())}
            for col in FEATURES
        },
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1_000_000
    print(f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)} ({size_mb:.1f} MB)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"selected_model": best_name, "results": results},
                                       indent=2))
    print(f"Saved {metrics_path.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
