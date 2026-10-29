from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "COLLEGE_MODEL (1).pkl"
DATA_PATH = PROJECT_ROOT / "JoSAA_2019_2024_CLEAN_FINAL.csv"

FEATURE_COLUMNS = [
    "institute",
    "academic_program_name",
    "quota",
    "seat_type",
    "gender",
    "year_norm",
    "round_norm",
    "institute_tier",
    "branch_demand",
    "home_advantage",
]
CATEGORICAL_FEATURE_INDICES = [0, 1, 2, 3, 4]
REQUIRED_DATA_COLUMNS = {
    "institute",
    "academic_program_name",
    "quota",
    "seat_type",
    "gender",
    "opening_rank",
    "closing_rank",
    "year",
    "round",
}


def load_model(path: str | Path = MODEL_PATH):
    """Load and validate the model artifact saved by the training notebook."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model artifact at {model_path}: {exc}") from exc

    if not isinstance(artifact, (tuple, list)) or len(artifact) != 3:
        raise ValueError("The model artifact must contain the model, feature columns, and categorical indices.")

    model, feature_columns, categorical_indices = artifact
    if list(feature_columns) != FEATURE_COLUMNS:
        raise ValueError("The model feature schema does not match the project's documented inference schema.")
    if list(categorical_indices) != CATEGORICAL_FEATURE_INDICES:
        raise ValueError("The model categorical feature indices do not match the training notebook.")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The serialized model does not provide a predict method.")

    return model


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same deterministic feature engineering used during training."""
    missing_columns = REQUIRED_DATA_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    engineered = data.copy()
    for column in ("institute", "academic_program_name", "quota", "seat_type", "gender"):
        if engineered[column].isna().any():
            raise ValueError(f"Dataset contains missing values in required column: {column}")

    for column in ("opening_rank", "closing_rank", "year", "round"):
        engineered[column] = pd.to_numeric(engineered[column], errors="raise")

    def institute_tier(name: str) -> int:
        if "Indian Institute of Technology" in name:
            return 1
        if "National Institute of Technology" in name:
            return 2
        if "Indian Institute of Information Technology" in name:
            return 3
        return 4

    def branch_demand(program: str) -> int:
        program = program.lower()
        if "computer" in program or "ai" in program or "data" in program:
            return 1
        if "elect" in program:
            return 2
        if "mechanical" in program or "civil" in program or "chemical" in program:
            return 3
        return 4

    engineered["institute_tier"] = engineered["institute"].map(institute_tier)
    engineered["branch_demand"] = engineered["academic_program_name"].map(branch_demand)
    engineered["home_advantage"] = (engineered["quota"] == "HS").astype(int)

    year_span = engineered["year"].max() - engineered["year"].min()
    engineered["year_norm"] = (
        (engineered["year"] - engineered["year"].min()) / year_span if year_span else 0.0
    )
    max_round = engineered["round"].max()
    engineered["round_norm"] = engineered["round"] / max_round if max_round else 0.0
    return engineered


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the repository's CSV and prepare the exact model feature columns."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Historical JoSAA dataset not found: {dataset_path}")
    return engineer_features(pd.read_csv(dataset_path))


def validate_rank(user_rank: object) -> int:
    if isinstance(user_rank, bool) or not isinstance(user_rank, Integral) or user_rank < 1:
        raise ValueError("Enter a whole-number JEE rank greater than zero.")
    return int(user_rank)


def _validate_choice(data: pd.DataFrame, column: str, value: object, label: str) -> None:
    if value not in set(data[column].unique()):
        raise ValueError(f"Choose a {label} available in the historical dataset.")


def recommend_colleges(
    data: pd.DataFrame,
    model,
    user_rank: object,
    year: int,
    round_no: int,
    seat_type: str,
    gender: str,
    quota: str,
    institutes: Sequence[str] | None = None,
    programs: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score matching historical rows and mark those within the model estimate."""
    rank = validate_rank(user_rank)
    _validate_choice(data, "year", year, "year")
    _validate_choice(data, "round", round_no, "round")
    _validate_choice(data, "seat_type", seat_type, "seat type")
    _validate_choice(data, "gender", gender, "gender category")
    _validate_choice(data, "quota", quota, "quota")

    if institutes:
        invalid = set(institutes).difference(data["institute"].unique())
        if invalid:
            raise ValueError("One or more selected institutes are not in the historical dataset.")
    if programs:
        invalid = set(programs).difference(data["academic_program_name"].unique())
        if invalid:
            raise ValueError("One or more selected programs are not in the historical dataset.")

    matches = data.loc[
        (data["year"] == year)
        & (data["round"] == round_no)
        & (data["seat_type"] == seat_type)
        & (data["gender"] == gender)
        & (data["quota"] == quota)
    ].copy()
    if institutes:
        matches = matches[matches["institute"].isin(institutes)]
    if programs:
        matches = matches[matches["academic_program_name"].isin(programs)]

    if matches.empty:
        matches["predicted_closing_rank"] = pd.Series(dtype="float64")
        matches["rank_difference"] = pd.Series(dtype="float64")
        matches["rank_eligible_by_estimate"] = pd.Series(dtype="bool")
        return matches

    predicted_ranks = model.predict(matches.loc[:, FEATURE_COLUMNS])
    matches["predicted_closing_rank"] = predicted_ranks
    matches["rank_difference"] = (matches["predicted_closing_rank"] - rank).abs()
    matches["rank_eligible_by_estimate"] = matches["predicted_closing_rank"] >= rank
    return matches