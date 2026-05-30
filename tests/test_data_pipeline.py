import pandas as pd

from src.data import (
    FEATURE_SET_ALEJANDRO,
    FEATURE_SET_VICTOR,
    build_modeling_dataset,
    clean_raw_data,
    engineer_features,
    engineer_features_alejandro,
    engineer_features_victor,
    get_feature_columns,
    split_features_target,
)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel", "City Hotel", "Resort Hotel"],
            "is_canceled": [0, 1, 0, 1],
            "lead_time": [10, 80, 20, 120],
            "arrival_date_year": [2015, 2016, 2016, 2017],
            "arrival_date_month": ["July", "December", "March", "January"],
            "arrival_date_week_number": [27, 52, 12, 2],
            "arrival_date_day_of_month": [1, 22, 15, 5],
            "stays_in_weekend_nights": [0, 2, 1, 1],
            "stays_in_week_nights": [2, 3, 2, 4],
            "adults": [2, 3, 1, 2],
            "children": [0.0, 1.0, None, 0.0],
            "babies": [0, 0, 1, 0],
            "meal": ["BB", "HB", "BB", "SC"],
            "country": ["PRT", "ESP", "FRA", None],
            "market_segment": ["Direct", "Online TA", "Groups", "Direct"],
            "distribution_channel": ["Direct", "TA/TO", "TA/TO", "Direct"],
            "is_repeated_guest": [0, 0, 1, 0],
            "previous_cancellations": [0, 1, 0, 0],
            "previous_bookings_not_canceled": [0, 0, 2, 0],
            "reserved_room_type": ["A", "D", "A", "C"],
            "assigned_room_type": ["A", "D", "A", "C"],
            "booking_changes": [0, 1, 0, 2],
            "deposit_type": ["No Deposit", "Non Refund", "No Deposit", "No Deposit"],
            "agent": [None, 9, 1, None],
            "company": [None, None, None, None],
            "days_in_waiting_list": [0, 0, 1, 0],
            "customer_type": ["Transient", "Transient", "Group", "Transient"],
            "adr": [90.0, 110.0, -5.0, 75.0],
            "required_car_parking_spaces": [0, 0, 1, 0],
            "total_of_special_requests": [0, 2, 1, 0],
            "reservation_status": ["Check-Out", "Canceled", "Check-Out", "Canceled"],
            "reservation_status_date": ["2015-07-01", "2016-12-01", "2016-03-10", "2017-01-01"],
        }
    )


def test_clean_raw_data_removes_leakage_columns_and_invalid_rows() -> None:
    cleaned = clean_raw_data(_raw_rows())

    assert len(cleaned) == 2
    assert cleaned["adr"].min() >= 0
    assert cleaned["children"].dtype.kind in {"i", "u"}
    assert "reservation_status" not in cleaned.columns
    assert "reservation_status_date" not in cleaned.columns
    assert "agent" not in cleaned.columns
    assert "company" not in cleaned.columns
    assert "arrival_date_year" not in cleaned.columns


def test_engineer_features_creates_business_columns_and_drops_sources() -> None:
    modeled = engineer_features_victor(clean_raw_data(_raw_rows()))

    expected_columns = {
        "adults_categories",
        "has_children",
        "has_babies",
        "arrival_season",
        "month_period",
        "arrival_quarter",
        "country_grouped",
    }

    assert expected_columns.issubset(modeled.columns)
    assert modeled.loc[modeled["lead_time"] == 10, "adults_categories"].item() == "2 adultos"
    assert modeled.loc[modeled["lead_time"] == 80, "adults_categories"].item() == "3 o mas adultos"
    assert modeled.loc[modeled["lead_time"] == 80, "has_children"].item() == 1
    assert modeled.loc[modeled["lead_time"] == 80, "arrival_season"].item() == "temporada_alta"
    assert modeled.loc[modeled["lead_time"] == 10, "arrival_quarter"].item() == "Q3"

    for source_column in [
        "adults",
        "children",
        "babies",
        "arrival_date_day_of_month",
        "arrival_date_week_number",
        "arrival_date_month",
        "country",
    ]:
        assert source_column not in modeled.columns


def test_split_features_target_keeps_binary_target_separate() -> None:
    modeled = engineer_features_victor(clean_raw_data(_raw_rows()))
    x_data, y_data = split_features_target(modeled)

    assert "is_canceled" not in x_data.columns
    assert y_data.tolist() == [0, 1]
    assert get_feature_columns(modeled) == list(x_data.columns)


def test_build_modeling_dataset_supports_victor_feature_set() -> None:
    modeled = build_modeling_dataset(_raw_rows(), feature_set=FEATURE_SET_VICTOR)

    assert "country_grouped" in modeled.columns
    assert "country" not in modeled.columns
    assert "arrival_date_week_number" not in modeled.columns
    assert "assigned_room_type" in modeled.columns
    assert modeled["is_canceled"].tolist() == [0, 1]


def test_alejandro_feature_set_keeps_granular_columns_and_adds_business_features() -> None:
    modeled = engineer_features_alejandro(clean_raw_data(_raw_rows()))

    expected_added_columns = {
        "adults_categories",
        "has_children",
        "has_babies",
        "arrival_season",
        "month_period",
        "arrival_quarter",
        "country_grouped",
        "total_nights",
        "total_guests",
        "has_special_requests",
        "has_parking_request",
        "adr_per_guest",
        "lead_time_bucket",
    }
    expected_granular_columns = {
        "adults",
        "children",
        "babies",
        "arrival_date_day_of_month",
        "arrival_date_week_number",
        "arrival_date_month",
        "country",
    }

    assert expected_added_columns.issubset(modeled.columns)
    assert expected_granular_columns.issubset(modeled.columns)
    assert "assigned_room_type" not in modeled.columns
    assert modeled.loc[modeled["lead_time"] == 80, "total_nights"].item() == 5
    assert modeled.loc[modeled["lead_time"] == 80, "total_guests"].item() == 4
    assert modeled.loc[modeled["lead_time"] == 80, "has_special_requests"].item() == 1
    assert modeled.loc[modeled["lead_time"] == 80, "has_parking_request"].item() == 0
    assert modeled.loc[modeled["lead_time"] == 10, "lead_time_bucket"].item() == "8-30"


def test_build_modeling_dataset_rejects_unknown_feature_set() -> None:
    try:
        build_modeling_dataset(_raw_rows(), feature_set="unknown")
    except ValueError as error:
        assert "feature_set" in str(error)
    else:
        raise AssertionError("Unknown feature_set should raise ValueError")


def test_build_modeling_dataset_supports_alejandro_feature_set() -> None:
    modeled = build_modeling_dataset(_raw_rows(), feature_set=FEATURE_SET_ALEJANDRO)

    assert "total_nights" in modeled.columns
    assert "assigned_room_type" not in modeled.columns
    assert "country" in modeled.columns
