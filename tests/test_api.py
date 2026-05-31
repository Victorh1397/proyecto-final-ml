from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier

from src.api import create_app
from src.data import FEATURE_SET_ALEJANDRO, build_modeling_dataset, split_features_target
from src.models import build_pipeline


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel", "City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel"],
            "is_canceled": [0, 1, 0, 1, 1, 0],
            "lead_time": [10, 80, 20, 120, 160, 5],
            "arrival_date_year": [2015, 2016, 2016, 2017, 2017, 2015],
            "arrival_date_month": ["July", "December", "March", "January", "August", "May"],
            "arrival_date_week_number": [27, 52, 12, 2, 32, 20],
            "arrival_date_day_of_month": [1, 22, 15, 5, 18, 9],
            "stays_in_weekend_nights": [0, 2, 1, 1, 2, 0],
            "stays_in_week_nights": [2, 3, 2, 4, 1, 1],
            "adults": [2, 3, 1, 2, 2, 1],
            "children": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            "babies": [0, 0, 1, 0, 0, 0],
            "meal": ["BB", "HB", "BB", "SC", "BB", "HB"],
            "country": ["PRT", "ESP", "FRA", "GBR", "PRT", "ESP"],
            "market_segment": ["Direct", "Online TA", "Groups", "Direct", "Online TA", "Direct"],
            "distribution_channel": ["Direct", "TA/TO", "TA/TO", "Direct", "TA/TO", "Direct"],
            "is_repeated_guest": [0, 0, 1, 0, 0, 1],
            "previous_cancellations": [0, 1, 0, 0, 1, 0],
            "previous_bookings_not_canceled": [0, 0, 2, 0, 0, 1],
            "reserved_room_type": ["A", "D", "A", "C", "D", "A"],
            "assigned_room_type": ["A", "D", "A", "C", "E", "A"],
            "booking_changes": [0, 1, 0, 2, 0, 1],
            "deposit_type": ["No Deposit", "Non Refund", "No Deposit", "No Deposit", "Non Refund", "No Deposit"],
            "agent": [None, 9, 1, None, 9, None],
            "company": [None, None, None, None, None, None],
            "days_in_waiting_list": [0, 0, 1, 0, 3, 0],
            "customer_type": ["Transient", "Transient", "Group", "Transient", "Transient", "Contract"],
            "adr": [90.0, 110.0, 75.0, 85.0, 130.0, 60.0],
            "required_car_parking_spaces": [0, 0, 1, 0, 0, 1],
            "total_of_special_requests": [0, 2, 1, 0, 1, 2],
            "reservation_status": ["Check-Out", "Canceled", "Check-Out", "Canceled", "Canceled", "Check-Out"],
            "reservation_status_date": [
                "2015-07-01",
                "2016-12-01",
                "2016-03-10",
                "2017-01-01",
                "2017-08-18",
                "2015-05-09",
            ],
        }
    )


def _write_model(models_dir: Path, name: str) -> None:
    modeled = build_modeling_dataset(_raw_rows(), feature_set=FEATURE_SET_ALEJANDRO)
    x_data, y_data = split_features_target(modeled)
    estimator = build_pipeline(DecisionTreeClassifier(max_depth=3, random_state=42))
    estimator.fit(x_data, y_data)
    joblib.dump(estimator, models_dir / f"{name}.joblib")


def test_models_endpoint_lists_available_joblib_models(tmp_path: Path) -> None:
    _write_model(tmp_path, "selected_tree")

    client = TestClient(create_app(models_dir=tmp_path))

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["selected_tree"]}


def test_predict_endpoint_uses_selected_model_and_feature_set(tmp_path: Path) -> None:
    _write_model(tmp_path, "selected_tree")
    client = TestClient(create_app(models_dir=tmp_path))
    record_frame = _raw_rows().drop(columns=["is_canceled"]).iloc[[0]].astype(object)
    record_frame = record_frame.where(pd.notna(record_frame), None)
    record = record_frame.to_dict(orient="records")[0]

    response = client.post(
        "/predict/selected_tree",
        json={"feature_set": FEATURE_SET_ALEJANDRO, "records": [record]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["model_name"] == "selected_tree"
    assert payload["feature_set"] == FEATURE_SET_ALEJANDRO
    assert payload["predictions"] in ([0], [1])
    assert len(payload["probabilities"]) == 1
    assert 0.0 <= payload["probabilities"][0] <= 1.0


def test_predict_endpoint_returns_404_for_unknown_model(tmp_path: Path) -> None:
    client = TestClient(create_app(models_dir=tmp_path))

    response = client.post("/predict/missing_model", json={"records": [{}]})

    assert response.status_code == 404
