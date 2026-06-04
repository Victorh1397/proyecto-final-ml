from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "is_canceled"
FEATURE_SET_VICTOR = "victor"
FEATURE_SET_ALEJANDRO = "alejandro"
FEATURE_SET_BINARY = "binary"
FEATURE_SET_POST_ASSIGNMENT = "post_asignacion"
FEATURE_SETS = (FEATURE_SET_VICTOR, FEATURE_SET_ALEJANDRO, FEATURE_SET_BINARY, FEATURE_SET_POST_ASSIGNMENT)

LEAKAGE_AND_ADMIN_COLUMNS = [
    "agent",
    "company",
    "reservation_status",
    "reservation_status_date",
    "arrival_date_year",
]

SOURCE_FEATURE_COLUMNS = [
    "adults",
    "children",
    "babies",
    "arrival_date_day_of_month",
    "arrival_date_week_number",
    "arrival_date_month",
    "country",
]

ASSIGNED_ROOM_COLUMN = "assigned_room_type"
BINARY_TOP_COUNTRY_COUNT = 15
COMMON_RESERVED_ROOM_TYPES = {"A", "D", "E", "F", "G"}
MONTH_TO_SEASON = {
    "December": "winter",
    "January": "winter",
    "February": "winter",
    "March": "spring",
    "April": "spring",
    "May": "spring",
    "June": "summer",
    "July": "summer",
    "August": "summer",
    "September": "autumn",
    "October": "autumn",
    "November": "autumn",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga el CSV de reservas hoteleras."""
    return pd.read_csv(path)


def clean_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset bruto evitando columnas administrativas o con fuga."""
    data = raw_data.copy()
    columns_to_drop = [column for column in LEAKAGE_AND_ADMIN_COLUMNS if column in data.columns]
    data = data.drop(columns=columns_to_drop)

    if "adr" in data.columns:
        data = data[data["adr"] >= 0]

    nullable_columns = [column for column in ["children", "country"] if column in data.columns]
    if nullable_columns:
        data = data.dropna(subset=nullable_columns)

    if "children" in data.columns:
        data["children"] = data["children"].astype("int64")

    return data.reset_index(drop=True)


def clean_raw_data_for_post_assignment_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Limpia para el modelo post-asignacion sin eliminar agent/company."""
    data = raw_data.copy()
    leakage_columns = ["reservation_status", "reservation_status_date", "arrival_date_year"]
    columns_to_drop = [column for column in leakage_columns if column in data.columns]
    data = data.drop(columns=columns_to_drop)

    if "adr" in data.columns:
        data = data[data["adr"] >= 0]

    nullable_columns = [column for column in ["children", "country"] if column in data.columns]
    if nullable_columns:
        data = data.dropna(subset=nullable_columns)

    if "children" in data.columns:
        data["children"] = data["children"].astype("int64")

    return data.reset_index(drop=True)


def assign_arrival_season(week_number: int | float) -> str:
    """Agrupa semanas en temporadas de negocio."""
    week = int(week_number)
    if (23 <= week <= 35) or week >= 51:
        return "temporada_alta"
    if (11 <= week <= 22) or (36 <= week <= 44):
        return "temporada_media"
    return "temporada_baja"


def assign_arrival_quarter(month: str) -> str:
    """Convierte el mes de llegada en trimestre."""
    if month in {"January", "February", "March"}:
        return "Q1"
    if month in {"April", "May", "June"}:
        return "Q2"
    if month in {"July", "August", "September"}:
        return "Q3"
    return "Q4"


def _add_calendar_guest_country_features(
    data: pd.DataFrame,
    top_country_count: int,
) -> pd.DataFrame:
    """Agrega variables derivadas compartidas por los feature sets."""
    modeled = data.copy()

    if "adults" in modeled.columns:
        adult_conditions = [
            modeled["adults"] <= 1,
            modeled["adults"] == 2,
            modeled["adults"] >= 3,
        ]
        adult_labels = ["1 adulto", "2 adultos", "3 o mas adultos"]
        modeled["adults_categories"] = np.select(adult_conditions, adult_labels, default="sin_adultos")

    if "children" in modeled.columns:
        modeled["has_children"] = (modeled["children"] > 0).astype("int64")

    if "babies" in modeled.columns:
        modeled["has_babies"] = (modeled["babies"] > 0).astype("int64")

    if "arrival_date_day_of_month" in modeled.columns:
        modeled["month_period"] = pd.cut(
            modeled["arrival_date_day_of_month"],
            bins=[0, 10, 20, 31],
            labels=["inicio_mes", "mitad_mes", "fin_mes"],
            include_lowest=True,
        ).astype("string")

    if "arrival_date_week_number" in modeled.columns:
        modeled["arrival_season"] = modeled["arrival_date_week_number"].apply(assign_arrival_season)

    if "arrival_date_month" in modeled.columns:
        modeled["arrival_quarter"] = modeled["arrival_date_month"].apply(assign_arrival_quarter)

    if "country" in modeled.columns:
        top_countries = modeled["country"].value_counts().nlargest(top_country_count).index
        modeled["country_grouped"] = modeled["country"].where(
            modeled["country"].isin(top_countries),
            "Rest_of_the_world",
        )

    return modeled.reset_index(drop=True)


def engineer_features_victor(data: pd.DataFrame, top_country_count: int = 15) -> pd.DataFrame:
    """Reproduce el dataset modelado originalmente por Victor."""
    modeled = _add_calendar_guest_country_features(data, top_country_count)
    columns_to_drop = [column for column in SOURCE_FEATURE_COLUMNS if column in modeled.columns]
    return modeled.drop(columns=columns_to_drop).reset_index(drop=True)


def engineer_features_alejandro(data: pd.DataFrame, top_country_count: int = 50) -> pd.DataFrame:
    """Mantiene granularidad y agrega variables de negocio sin usar habitacion asignada."""
    modeled = _add_calendar_guest_country_features(data, top_country_count)

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(modeled.columns):
        modeled["total_nights"] = modeled["stays_in_weekend_nights"] + modeled["stays_in_week_nights"]

    if {"adults", "children", "babies"}.issubset(modeled.columns):
        modeled["total_guests"] = modeled["adults"] + modeled["children"] + modeled["babies"]
        denominator = modeled["total_guests"].replace(0, np.nan)
        modeled["adr_per_guest"] = (modeled["adr"] / denominator).replace([np.inf, -np.inf], np.nan)
        modeled["adr_per_guest"] = modeled["adr_per_guest"].fillna(modeled["adr"])

    if "total_of_special_requests" in modeled.columns:
        modeled["has_special_requests"] = (modeled["total_of_special_requests"] > 0).astype("int64")

    if "required_car_parking_spaces" in modeled.columns:
        modeled["has_parking_request"] = (modeled["required_car_parking_spaces"] > 0).astype("int64")

    if "lead_time" in modeled.columns:
        modeled["lead_time_bucket"] = pd.cut(
            modeled["lead_time"],
            bins=[-1, 7, 30, 90, 180, 400, 1000],
            labels=["0-7", "8-30", "31-90", "91-180", "181-400", "400+"],
        ).astype("string")

    if ASSIGNED_ROOM_COLUMN in modeled.columns:
        modeled = modeled.drop(columns=[ASSIGNED_ROOM_COLUMN])

    return modeled.reset_index(drop=True)


def engineer_features_binary(data: pd.DataFrame, top_country_count: int = BINARY_TOP_COUNTRY_COUNT) -> pd.DataFrame:
    """Crea una variante experimental que conserva granularidad y agrega indicadores."""
    modeled = engineer_features_alejandro(data, top_country_count)

    if {"adr", "total_guests"}.issubset(modeled.columns):
        modeled = modeled[
            (modeled["adr"] <= 1000)
            & (modeled["total_guests"] > 0)
            & (modeled["total_guests"] <= 10)
        ].copy()

    if "previous_cancellations" in modeled.columns:
        modeled["has_previous_cancellations"] = (modeled["previous_cancellations"] > 0).astype("int64")

    if "previous_bookings_not_canceled" in modeled.columns:
        modeled["has_previous_bookings"] = (modeled["previous_bookings_not_canceled"] > 0).astype("int64")

    if "days_in_waiting_list" in modeled.columns:
        modeled["has_waiting_days"] = (modeled["days_in_waiting_list"] > 0).astype("int64")

    if "stays_in_weekend_nights" in modeled.columns:
        modeled["is_weekend_stay"] = (modeled["stays_in_weekend_nights"] > 0).astype("int64")

    if "total_nights" in modeled.columns:
        modeled["is_long_stay"] = (modeled["total_nights"] >= 7).astype("int64")

    if "adr" in modeled.columns:
        high_adr_threshold = modeled["adr"].quantile(0.75)
        modeled["is_high_adr"] = (modeled["adr"] >= high_adr_threshold).astype("int64")

    if "lead_time" in modeled.columns:
        modeled["is_last_minute_booking"] = (modeled["lead_time"] <= 7).astype("int64")
        modeled["is_early_booking"] = (modeled["lead_time"] >= 90).astype("int64")

    if "deposit_type" in modeled.columns:
        modeled["is_non_refundable"] = (modeled["deposit_type"] == "Non Refund").astype("int64")

    if "country" in modeled.columns:
        modeled["is_portugal"] = (modeled["country"] == "PRT").astype("int64")
        modeled = modeled.drop(columns=["country"])

    if "market_segment" in modeled.columns:
        modeled["is_online_ta"] = (modeled["market_segment"] == "Online TA").astype("int64")
        modeled["is_group_segment"] = (modeled["market_segment"] == "Groups").astype("int64")

    if "reserved_room_type" in modeled.columns:
        modeled["reserved_room_type"] = modeled["reserved_room_type"].where(
            modeled["reserved_room_type"].isin(COMMON_RESERVED_ROOM_TYPES),
            "Other",
        )

    return modeled.reset_index(drop=True)


def engineer_features_post_assignment(data: pd.DataFrame) -> pd.DataFrame:
    """Crea variables disponibles despues de asignar habitacion."""
    modeled = data.copy()

    if {"adults", "children", "babies"}.issubset(modeled.columns):
        modeled["total_guests"] = modeled["adults"] + modeled["children"] + modeled["babies"]
        denominator = modeled["total_guests"].replace(0, np.nan)
        modeled["adr_per_person"] = (modeled["adr"] / denominator).replace([np.inf, -np.inf], np.nan)
        modeled["adr_per_person"] = modeled["adr_per_person"].fillna(modeled["adr"])

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(modeled.columns):
        modeled["total_nights"] = modeled["stays_in_weekend_nights"] + modeled["stays_in_week_nights"]

    if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(modeled.columns):
        previous_total = modeled["previous_cancellations"] + modeled["previous_bookings_not_canceled"]
        denominator = previous_total.replace(0, np.nan)
        modeled["previous_cancel_ratio"] = (
            modeled["previous_cancellations"] / denominator
        ).replace([np.inf, -np.inf], np.nan)
        modeled["previous_cancel_ratio"] = modeled["previous_cancel_ratio"].fillna(0.0)

    if {"assigned_room_type", "reserved_room_type"}.issubset(modeled.columns):
        modeled["room_changed"] = (modeled["assigned_room_type"] != modeled["reserved_room_type"]).astype("int64")
        modeled = modeled.drop(columns=["assigned_room_type"])

    if "agent" in modeled.columns:
        modeled["has_agent"] = modeled["agent"].notna().astype("int64")
        modeled = modeled.drop(columns=["agent"])

    if "company" in modeled.columns:
        modeled["has_company"] = modeled["company"].notna().astype("int64")
        modeled = modeled.drop(columns=["company"])

    if "arrival_date_month" in modeled.columns:
        modeled["arrival_season"] = modeled["arrival_date_month"].map(MONTH_TO_SEASON).fillna("unknown")

    return modeled.reset_index(drop=True)


def engineer_features(data: pd.DataFrame, top_country_count: int = 15) -> pd.DataFrame:
    """Compatibilidad hacia atras: usa el feature set de Victor."""
    return engineer_features_victor(data, top_country_count)


def build_modeling_dataset(raw_data: pd.DataFrame, feature_set: str = FEATURE_SET_VICTOR) -> pd.DataFrame:
    """Ejecuta limpieza e ingenieria de variables desde el dataset bruto."""
    if feature_set == FEATURE_SET_POST_ASSIGNMENT:
        return engineer_features_post_assignment(clean_raw_data_for_post_assignment_features(raw_data))

    cleaned = clean_raw_data(raw_data)
    if feature_set == FEATURE_SET_VICTOR:
        return engineer_features_victor(cleaned)
    if feature_set == FEATURE_SET_ALEJANDRO:
        return engineer_features_alejandro(cleaned)
    if feature_set == FEATURE_SET_BINARY:
        return engineer_features_binary(cleaned)
    raise ValueError(f"feature_set debe ser uno de: {', '.join(FEATURE_SETS)}.")


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y objetivo."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"El dataset debe contener la columna objetivo '{TARGET_COLUMN}'.")
    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN].astype(int)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Devuelve las columnas predictoras en orden estable."""
    return [column for column in data.columns if column != TARGET_COLUMN]
