import pytest

from wandb_cache.filters import matches_filter

RECORD = {
    "run_id": "abc123",
    "run_name": "hopper-seed-1",
    "run_state": "finished",
    "run_tags": ["pr-424", "sac"],
    "config": {
        "env_id": "Hopper-v4",
        "exp_name": "sac_continuous_action",
        "seed": 1,
    },
}


def test_matches_filter_accepts_empty_filters() -> None:
    assert matches_filter(RECORD, None)
    assert matches_filter(RECORD, {})


def test_matches_filter_supports_aliases_nested_fields_and_tags() -> None:
    assert matches_filter(
        RECORD,
        {
            "id": "abc123",
            "display_name": {"$regex": "hopper"},
            "tags": "pr-424",
            "config.env_id": "Hopper-v4",
        },
    )


def test_matches_filter_supports_logical_operators() -> None:
    assert matches_filter(
        RECORD,
        {
            "$and": [
                {"config.seed": {"$gte": 1}},
                {"$or": [{"config.env_id": "Walker2d-v4"}, {"config.env_id": "Hopper-v4"}]},
            ]
        },
    )


def test_matches_filter_supports_in_and_nin() -> None:
    assert matches_filter(RECORD, {"config.env_id": {"$in": ["Hopper-v4", "HalfCheetah-v4"]}})
    assert matches_filter(RECORD, {"tags": {"$in": ["benchmark", "sac"]}})
    assert matches_filter(RECORD, {"config.env_id": {"$nin": ["Walker2d-v4"]}})


def test_matches_filter_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        matches_filter(RECORD, {"config.seed": {"$contains": 1}})
