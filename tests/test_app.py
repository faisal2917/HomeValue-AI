"""Drive the Streamlit app headlessly with different inputs."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def number_input(at, label):
    return next(w for w in at.number_input if w.label == label)


def click_predict(at):
    at.button[0].click().run()
    assert not at.exception
    return at


def estimate(at):
    """Return the price shown in the result card, e.g. '$286,912', or None if absent."""
    match = next((re.search(r'class="result-value">([^<]+)<', m.value)
                  for m in at.markdown if 'class="result-value"' in m.value), None)
    return match.group(1) if match else None


def test_default_inputs_produce_a_price():
    at = click_predict(run_app())
    assert estimate(at).startswith("$")
    assert not at.error


@pytest.mark.parametrize("city", ["San Francisco", "Fresno", "San Diego", "Redding"])
def test_each_city_produces_a_price(city):
    at = run_app()
    at.selectbox[0].select(city).run()
    at = click_predict(at)
    assert estimate(at).startswith("$")


def test_custom_coordinates_are_used():
    at = run_app()
    at.selectbox[0].select("Custom coordinates…").run()
    number_input(at, "Latitude").set_value(34.05)
    number_input(at, "Longitude").set_value(-118.24)
    at = click_predict(at)
    assert estimate(at).startswith("$")


def test_more_bedrooms_than_rooms_shows_an_error():
    at = run_app()
    number_input(at, "Average rooms").set_value(2.0)
    number_input(at, "Average bedrooms").set_value(4.0)
    at = click_predict(at)
    assert any("bedrooms cannot exceed" in e.value.lower() for e in at.error)
    assert estimate(at) is None


def test_income_above_training_range_warns():
    at = run_app()
    number_input(at, "Median household income ($/year)").set_value(250_000)
    at = click_predict(at)
    assert estimate(at).startswith("$")
    assert any("income" in w.value.lower() for w in at.warning)
