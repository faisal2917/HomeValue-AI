import pytest

from homevalue.predict import format_currency, load_artifact, predict_price, validate_inputs


@pytest.fixture(scope="module")
def artifact():
    return load_artifact()


def make_values(**overrides):
    values = {
        "MedInc": 6.0, "HouseAge": 25.0, "AveRooms": 5.0, "AveBedrms": 1.0,
        "Population": 1200.0, "AveOccup": 2.8, "Latitude": 37.77, "Longitude": -122.42,
    }
    values.update(overrides)
    return values


def test_valid_input_has_no_errors(artifact):
    errors, warnings = validate_inputs(make_values(), artifact["feature_ranges"])
    assert errors == []
    assert warnings == []


@pytest.mark.parametrize("overrides, fragment", [
    ({"AveBedrms": 6.0}, "bedrooms cannot exceed"),
    ({"MedInc": 0.0}, "income must be greater than zero"),
    ({"Latitude": 45.0}, "Latitude must be within California"),
    ({"Longitude": -100.0}, "Longitude must be within California"),
    ({"HouseAge": -1.0}, "cannot be negative"),
])
def test_invalid_input_is_rejected(artifact, overrides, fragment):
    errors, _ = validate_inputs(make_values(**overrides), artifact["feature_ranges"])
    assert any(fragment.lower() in e.lower() for e in errors)


def test_missing_value_is_rejected(artifact):
    values = make_values()
    del values["Population"]
    errors, _ = validate_inputs(values, artifact["feature_ranges"])
    assert errors and "Neighborhood population" in errors[0]


def test_out_of_training_range_warns(artifact):
    errors, warnings = validate_inputs(make_values(MedInc=30.0), artifact["feature_ranges"])
    assert errors == []
    assert any("income" in w.lower() for w in warnings)


def test_prediction_is_a_plausible_dollar_amount(artifact):
    price = predict_price(artifact, make_values())
    assert 15_000 <= price <= 500_001


def test_higher_income_predicts_higher_value(artifact):
    low = predict_price(artifact, make_values(MedInc=2.0))
    high = predict_price(artifact, make_values(MedInc=10.0))
    assert high > low


def test_coastal_is_pricier_than_central_valley(artifact):
    san_francisco = predict_price(artifact, make_values(Latitude=37.77, Longitude=-122.42))
    fresno = predict_price(artifact, make_values(Latitude=36.74, Longitude=-119.79))
    assert san_francisco > fresno


def test_format_currency():
    assert format_currency(234567.89) == "$234,568"
    assert format_currency(0) == "$0"
