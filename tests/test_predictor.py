from pathlib import Path

import numpy as np
import pytest

from predictor import (
    CATEGORICAL_FEATURE_INDICES,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    load_dataset,
    load_model,
    recommend_colleges,
    validate_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


def test_model_loads_with_expected_feature_schema(model):
    assert model.feature_names_ == FEATURE_COLUMNS
    assert model.get_cat_feature_indices() == CATEGORICAL_FEATURE_INDICES


def test_real_model_predicts_for_valid_inputs_and_keeps_historical_ranks(dataset, model):
    results = recommend_colleges(
        dataset,
        model,
        user_rank=10_000,
        year=2024,
        round_no=5,
        quota="AI",
        seat_type="OPEN",
        gender="Gender-Neutral",
    )

    assert len(results) == 549
    assert np.isfinite(results["predicted_closing_rank"]).all()
    assert results["closing_rank"].equals(
        dataset.loc[results.index, "closing_rank"]
    )
    assert results["rank_eligible_by_estimate"].equals(
        results["predicted_closing_rank"] >= 10_000
    )


@pytest.mark.parametrize("rank", [0, -1, 1.5, "100", None, True])
def test_invalid_ranks_are_rejected(rank):
    with pytest.raises(ValueError, match="whole-number JEE rank"):
        validate_rank(rank)


def test_valid_rank_is_normalized_to_python_integer():
    assert validate_rank(np.int64(12_345)) == 12_345


def test_missing_model_artifact_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(tmp_path / "missing.pkl")


def test_invalid_filter_choices_are_rejected(dataset, model):
    with pytest.raises(ValueError, match="quota available"):
        recommend_colleges(
            dataset,
            model,
            user_rank=1,
            year=2024,
            round_no=5,
            quota="NOT-A-QUOTA",
            seat_type="OPEN",
            gender="Gender-Neutral",
        )


def test_valid_filters_can_return_no_historical_records(dataset, model):
    year_rows = dataset[(dataset["year"] == 2024) & (dataset["round"] == 5)]
    default_rows = year_rows[
        (year_rows["quota"] == "AI")
        & (year_rows["seat_type"] == "OPEN")
        & (year_rows["gender"] == "Gender-Neutral")
    ]
    absent_institute = next(
        institute
        for institute in year_rows["institute"].unique()
        if institute not in set(default_rows["institute"])
    )

    results = recommend_colleges(
        dataset,
        model,
        user_rank=10_000,
        year=2024,
        round_no=5,
        quota="AI",
        seat_type="OPEN",
        gender="Gender-Neutral",
        institutes=[absent_institute],
    )

    assert results.empty


def test_dataset_covers_only_years_and_rounds_present_in_the_csv(dataset):
    assert sorted(dataset["year"].unique().tolist()) == [2021, 2022, 2023, 2024]
    assert set(dataset.loc[dataset["year"] == 2024, "round"]) == {1, 5}
    assert {"opening_rank", "closing_rank"}.issubset(dataset.columns)