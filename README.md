# HomeValue AI

Predict the median house value of a California neighborhood from census data, using
scikit-learn regression models and an interactive Streamlit app.

## How it works

The project has two halves. An **offline training pipeline** runs once, picks the best
model, and saves it to a file. The **Streamlit app** loads that file and turns user input
into a price estimate.

```mermaid
flowchart TB
    subgraph OFFLINE["Offline: run once"]
        direction TB
        DS[("California Housing dataset<br/>20,640 neighborhoods")]
        EX["scripts/explore.py<br/>statistics and charts"]
        TR["scripts/train.py<br/>train and compare models"]
        DS --> EX
        DS --> TR
    end

    MODEL[("models/<br/>house_price_model.joblib")]

    subgraph ONLINE["Online: Streamlit app"]
        direction LR
        APP["app.py<br/>input form and result card"]
        USER(["Visitor"])
        USER -- "property details" --> APP
        APP -- "estimated price" --> USER
    end

    TR -- "save best model" --> MODEL
    MODEL -- "load once, cached" --> APP
    EX -. "reports/figures" .-> REPORTS[("reports/")]
    TR -. "metrics.json" .-> REPORTS
```

### Training pipeline (`scripts/train.py`)

```mermaid
flowchart TD
    A["Load data<br/>fetch_california_housing()"] --> B["Separate features X<br/>and target y"]
    B --> C["Split 80% train / 20% test<br/>random_state = 42"]
    C --> D1["Linear Regression<br/>+ StandardScaler"]
    C --> D2["Decision Tree<br/>max_depth = 12"]
    C --> D3["Random Forest<br/>200 trees"]
    D1 --> E["Evaluate each model on the unseen test set<br/>MAE · RMSE · R²"]
    D2 --> E
    D3 --> E
    E --> F{"Lowest test RMSE?"}
    F -- "Random Forest wins" --> G["Refit the winner on all 20,640 rows"]
    G --> H[("Save with joblib<br/>pipeline + metrics + feature ranges")]
    E --> I[("reports/metrics.json")]
```

### What happens when you click **Predict value** (`app.py`)

```mermaid
flowchart TD
    U(["Visitor picks a location and<br/>enters neighborhood details"]) --> P["Click Predict value"]
    P --> V{"validate_inputs()"}
    V -- "Invalid: e.g. bedrooms > rooms,<br/>location outside California" --> ERR["Show error message<br/>no prediction made"]
    V -- "Valid" --> R{"Inside the range seen<br/>during training?"}
    R -- "No" --> W["Add a reliability warning"]
    R -- "Yes" --> M
    W --> M["predict_price()<br/>Random Forest pipeline"]
    M --> OUT["Result card:<br/>price · likely range ± MAE ·<br/>vs. California median · map"]
```

## Project layout

```
app.py                     Streamlit UI (Phase 5)
assets/style.css           App styling
.streamlit/config.toml     Streamlit theme and settings (no secrets)
homevalue/data.py          Dataset loading, feature list, train/test split
homevalue/predict.py       Model loading, input validation, prediction, currency formatting
scripts/explore.py         Phase 1: inspect the data, save charts to reports/figures/
scripts/train.py           Phases 2-4: prepare, train, evaluate, save the best model
models/house_price_model.joblib   Saved model (pipeline + metadata)
reports/metrics.json       Train/test metrics for every model
tests/                     Unit tests and headless Streamlit app tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Phase 1: Dataset and exploration

```bash
python scripts/explore.py
```

`fetch_california_housing()` returns 20,640 rows, one per **census block group** (a
neighborhood, not a single house), with 8 numeric features and no missing values:

| Feature | Meaning |
|---|---|
| `MedInc` | Median household income, in $10,000s |
| `HouseAge` | Median house age, in years |
| `AveRooms` / `AveBedrms` | Average rooms / bedrooms per household |
| `Population` | People in the block group |
| `AveOccup` | Average people per household |
| `Latitude` / `Longitude` | Location |

**Target:** `MedHouseVal`, the median house value in $100,000s (1990 census). It is capped at
5.00001 ($500,001), and about 4.8% of rows sit at that cap.

Findings: median income is the strongest predictor (correlation 0.69). Location matters a
lot too, with coastal areas around the Bay Area and Los Angeles the most expensive (see
`reports/figures/geo_value_map.png`). `AveRooms`, `AveBedrms` and `AveOccup` have extreme
outliers.

## Phases 2-4: Prepare, train, evaluate

```bash
python scripts/train.py
```

- **Preparation:** split features `X` from target `y`, then hold out 20% as a test set the
  models never see during training. Test performance estimates how the model will do on new
  neighborhoods. Training performance alone rewards memorizing the data.
- **Scaling:** `LinearRegression` gets a `StandardScaler` in its pipeline. Tree models split on
  thresholds, so scaling does not affect them.
- **Models:** each model learns a function from the 8 features to a continuous number.
  Linear regression fits a weighted sum. A decision tree learns if/then splits and predicts
  the average of the training rows in a leaf. A random forest averages 200 trees, each
  trained on a random sample of the data.

Results on the unseen test set:

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| Linear Regression | 0.613 | 0.576 | $53,320 | $74,558 |
| Decision Tree | 0.838 | 0.723 | $40,483 | $60,257 |
| **Random Forest** | 0.941 | **0.804** | **$32,771** | **$50,679** |

- **MAE**: the average size of an error in dollars.
- **RMSE**: like MAE, but large misses count for more.
- **R²**: the share of variance in prices the model explains (1.0 is perfect).

The script picks the model with the lowest test RMSE (Random Forest), refits it on all
20,640 rows, and saves it with `joblib`.

## Phase 5: Streamlit app

```bash
streamlit run app.py
```

Pick a city (or enter custom coordinates), fill in the neighborhood details, and click
**Predict value**. The result card shows the estimate in dollars, a likely range of ± the
model's MAE, a comparison with the California median, and the location on a map. Styling
lives in `assets/style.css` and the theme in `.streamlit/config.toml`.
It blocks impossible inputs, such as more bedrooms than rooms or a location outside
California, and warns when an input is outside the range the model was trained on.

## Phase 6: Save, test, deploy

The model is saved by `scripts/train.py` and loaded once in the app with
`st.cache_resource`. Run the tests:

```bash
pytest
```

`tests/test_app.py` drives the real app headlessly with Streamlit's `AppTest`. It covers
several cities, custom coordinates, invalid input and out-of-range warnings.

### Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository, including `models/house_price_model.joblib`
   (about 33 MB, under GitHub's 100 MB file limit).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and choose
   the repository, branch, and `app.py` as the main file.
3. Under **Advanced settings**, choose Python 3.12 (or 3.9, which this project was built
   with). Streamlit Cloud installs `requirements.txt` automatically.
4. Click **Deploy**.

`requirements.txt` pins `scikit-learn==1.6.1` because a joblib model should be loaded with
the same scikit-learn version that trained it. If you upgrade scikit-learn, rerun
`scripts/train.py`.

### Secrets

The app currently needs no API keys or passwords. If you add any, put them in
`.streamlit/secrets.toml` locally (it is git-ignored) and in the app's **Settings → Secrets**
on Streamlit Community Cloud. Never commit them: this repository is public, and
`.gitignore` already excludes `.env` files, key files and credential JSON files.
