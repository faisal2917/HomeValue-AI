"""Phase 5: Streamlit UI for HomeValue AI.

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from homevalue.data import REPORTS_DIR
from homevalue.predict import (
    PRICE_CAP_DOLLARS,
    STATE_MEDIAN_DOLLARS,
    format_currency,
    load_artifact,
    predict_price,
    validate_inputs,
)

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "style.css"

# Approximate city-centre coordinates offered as a shortcut for latitude/longitude.
LOCATIONS = {
    "Los Angeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89),
    "Oakland": (37.80, -122.27),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
    "Bakersfield": (35.37, -119.02),
    "Riverside": (33.95, -117.40),
    "Santa Barbara": (34.42, -119.70),
    "Redding": (40.59, -122.39),
}
CUSTOM_LOCATION = "Custom coordinates…"

st.set_page_config(page_title="HomeValue AI · California Home Value Estimator",
                   page_icon="🏠", layout="wide")
st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_artifact():
    return load_artifact()


@st.cache_data
def get_model_comparison():
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())["results"]
    return pd.DataFrame([
        {
            "Model": name,
            "Test R²": r["test"]["R2"],
            "Test MAE": format_currency(r["test"]["MAE_dollars"]),
            "Test RMSE": format_currency(r["test"]["RMSE_dollars"]),
        }
        for name, r in results.items()
    ])


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title"><span class="dot"></span>{text}</div>',
                unsafe_allow_html=True)


def panel_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f"""
<div class="panel-head">
  <div class="panel-icon">{icon}</div>
  <div><div class="panel-title">{title}</div><div class="panel-sub">{subtitle}</div></div>
</div>
""", unsafe_allow_html=True)


def render_hero(artifact: dict) -> None:
    m = artifact["test_metrics"]
    st.markdown(f"""
<div class="hero">
  <span class="hero-badge">California housing · Machine learning valuation</span>
  <div class="hero-title">HomeValue AI</div>
  <p class="hero-sub">Estimate the median home value of any California neighborhood in
  seconds, using a {artifact['model_name']} model trained on 20,640 census block groups.</p>
  <div class="hero-stats">
    <div class="hero-stat"><div class="hero-stat-value">{m['R2']:.2f}</div>
      <div class="hero-stat-label">R² on unseen data</div></div>
    <div class="hero-stat"><div class="hero-stat-value">{format_currency(m['MAE_dollars'])}</div>
      <div class="hero-stat-label">Typical error (MAE)</div></div>
    <div class="hero-stat"><div class="hero-stat-value">20,640</div>
      <div class="hero-stat-label">Neighborhoods analysed</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_result(price: float, mae: float, place: str) -> None:
    low, high = max(price - mae, 0), price + mae
    pct = lambda dollars: min(dollars / PRICE_CAP_DOLLARS, 1) * 100  # noqa: E731
    vs_state = (price - STATE_MEDIAN_DOLLARS) / STATE_MEDIAN_DOLLARS
    trend = "up" if vs_state >= 0 else "down"
    st.markdown(f"""
<div class="result-card">
  <div class="result-label">Estimated median home value</div>
  <div class="result-value">{format_currency(price)}</div>
  <div class="result-range">Likely range <strong>{format_currency(low)}</strong> –
    <strong>{format_currency(high)}</strong></div>
  <div class="range-track">
    <div class="range-fill" style="left:{pct(low):.1f}%;width:{pct(high) - pct(low):.1f}%"></div>
    <div class="range-marker" style="left:{pct(price):.1f}%"></div>
  </div>
  <div class="range-scale"><span>$0</span><span>$250k</span><span>$500k+</span></div>
  <div class="chips">
    <span class="chip">📍 {place}</span>
    <span class="chip {trend}">{vs_state:+.0%} vs. California median</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_placeholder() -> None:
    st.markdown("""
<div class="empty-card">
  <div class="empty-icon">🏡</div>
  <div class="empty-title">Your estimate will appear here</div>
  <div class="empty-sub">Choose a location, describe the neighborhood, then click
  <b>Predict value</b>.</div>
</div>
""", unsafe_allow_html=True)


try:
    artifact = get_artifact()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

render_hero(artifact)

form_col, result_col = st.columns([1.15, 1], gap="large")

# Panels and the sections inside them are keyed containers; assets/style.css styles them
# through the st-key-<key> class Streamlit adds.
with form_col, st.container(border=True, key="input_panel"):
    panel_header("🏠", "Property details", "Describe the neighborhood you want to value")

    with st.container(border=True, key="sec_location"):
        section_title("Location")
        # Kept outside the form so choosing "Custom coordinates" immediately reveals the fields.
        location = st.selectbox("City or area", [*LOCATIONS, CUSTOM_LOCATION],
                                help="Pick a city, or enter exact coordinates.")

    with st.form("property_form", border=False):
        if location == CUSTOM_LOCATION:
            with st.container(border=True, key="sec_coordinates"):
                section_title("Coordinates")
                lat_col, lon_col = st.columns(2)
                lat = lat_col.number_input("Latitude", min_value=32.5, max_value=42.0,
                                           value=36.78, step=0.01, format="%.2f")
                lon = lon_col.number_input("Longitude", min_value=-124.5, max_value=-114.1,
                                           value=-119.42, step=0.01, format="%.2f")
        else:
            lat, lon = LOCATIONS[location]

        neighborhood = st.container(border=True, key="sec_neighborhood")
        with neighborhood:
            section_title("Neighborhood")
        col1, col2 = neighborhood.columns(2)
        income = col1.number_input("Median household income ($/year)", min_value=5_000,
                                   max_value=500_000, value=60_000, step=1_000,
                                   help="Training data tops out at $150,000.")
        house_age = col2.number_input("Median house age (years)", min_value=0, max_value=100,
                                      value=25, step=1)
        population = col1.number_input("Neighborhood population", min_value=1,
                                       max_value=50_000, value=1_200, step=50,
                                       help="People in the census block group "
                                            "(typically 600–3,000).")
        occupants = col2.number_input("Average occupants per home", min_value=0.5,
                                      max_value=20.0, value=2.8, step=0.1, format="%.1f")

        home = st.container(border=True, key="sec_home")
        with home:
            section_title("Typical home")
        col3, col4 = home.columns(2)
        rooms = col3.number_input("Average rooms", min_value=1.0, max_value=20.0, value=5.0,
                                  step=0.5, format="%.1f")
        bedrooms = col4.number_input("Average bedrooms", min_value=0.5, max_value=10.0,
                                     value=1.0, step=0.5, format="%.1f")

        submitted = st.form_submit_button("Predict value", type="primary",
                                          icon=":material/insights:", use_container_width=True)

with result_col, st.container(border=True, key="output_panel"):
    panel_header("📈", "Valuation result", "Model estimate with its likely range")
    if not submitted:
        render_placeholder()
    else:
        values = {
            "MedInc": income / 10_000,  # the model expects income in units of $10,000
            "HouseAge": float(house_age),
            "AveRooms": rooms,
            "AveBedrms": bedrooms,
            "Population": float(population),
            "AveOccup": occupants,
            "Latitude": lat,
            "Longitude": lon,
        }
        errors, warnings = validate_inputs(values, artifact["feature_ranges"])

        if errors:
            render_placeholder()
            for message in errors:
                st.error(message, icon=":material/error:")
        else:
            price = predict_price(artifact, values)
            place = location if location != CUSTOM_LOCATION else f"{lat:.2f}, {lon:.2f}"
            render_result(price, artifact["test_metrics"]["MAE_dollars"], place)
            if price >= 0.95 * PRICE_CAP_DOLLARS:
                st.info("This estimate is near the $500,001 cap in the training data; "
                        "the true value may be higher.", icon=":material/info:")
            for message in warnings:
                st.warning(message, icon=":material/warning:")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=8, size=2500,
                   color="#0F766E", height=260)

st.write("")
with st.expander("How the model works", icon=":material/model_training:"):
    st.markdown(
        "Three regression models were trained on 80% of the 1990 California census data and "
        "evaluated on the remaining 20%, which they never saw during training. The model with "
        "the lowest test RMSE is used for predictions."
    )
    comparison = get_model_comparison()
    if comparison is not None:
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("**MAE**: average size of an error. **RMSE**: like MAE, but large misses count "
               "for more. **R²**: share of price variation explained (1.0 is perfect).")

st.markdown(
    '<div class="app-footer">Estimates use 1990 census data and are capped at $500,001. '
    "For educational purposes only, not a professional appraisal.</div>",
    unsafe_allow_html=True,
)
