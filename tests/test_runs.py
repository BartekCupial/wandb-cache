from pathlib import Path
from typing import Any

from wandb_cache import WandbRunCache

RUN_RECORDS = [
    {
        "run_id": "run-1",
        "run_name": "hopper-seed-1",
        "run_state": "finished",
        "run_group": None,
        "run_tags": ["pr-424", "sac"],
        "run_created_at": "2026-05-21T00:00:00Z",
        "config": {
            "env_id": "Hopper-v4",
            "exp_name": "sac_continuous_action",
            "seed": 1,
            "optimizer.lr": 0.001,
            "llm_actor": {"engine_args": {"model_id": "test-model"}},
        },
    },
    {
        "run_id": "run-2",
        "run_name": "walker-seed-2",
        "run_state": "finished",
        "run_group": None,
        "run_tags": ["pr-424", "sac"],
        "run_created_at": "2026-05-21T00:00:01Z",
        "config": {
            "env_id": "Walker2d-v4",
            "exp_name": "sac_continuous_action",
            "seed": 2,
            "optimizer.lr": 0.002,
            "llm_actor": {"engine_args": {"model_id": "test-model"}},
        },
    },
]


def test_dataframe_omits_config_columns_by_default(tmp_path: Path) -> None:
    cache = cache_with_metadata(tmp_path)

    df = cache.dataframe(refresh_cache=True)

    assert len(df) == 2
    assert "config.env_id" not in df.columns
    assert "config.seed" not in df.columns
    assert "config" not in df.columns


def test_dataframe_includes_selected_config_keys(tmp_path: Path) -> None:
    cache = cache_with_metadata(tmp_path)

    df = cache.dataframe(
        refresh_cache=True,
        config_keys=[
            "env_id",
            "config.llm_actor.engine_args.model_id",
            "optimizer.lr",
            "missing",
        ],
    )

    assert df["config.env_id"].tolist() == ["Hopper-v4", "Walker2d-v4"]
    assert df["config.llm_actor.engine_args.model_id"].tolist() == ["test-model", "test-model"]
    assert df["config.optimizer.lr"].tolist() == [0.001, 0.002]
    assert "config.seed" not in df.columns
    assert "config.missing" not in df.columns


def test_records_keep_full_config_in_metadata_cache(tmp_path: Path) -> None:
    cache = cache_with_metadata(tmp_path)

    cache.dataframe(refresh_cache=True, config_keys=["env_id"])
    payload = cache.store.load()

    assert payload["records"][0]["config"]["seed"] == 1
    assert payload["records"][0]["config"]["llm_actor"]["engine_args"]["model_id"] == "test-model"


def test_table_dataframe_filters_config_columns_from_cached_rows(tmp_path: Path) -> None:
    cache = WandbRunCache(project="entity/project", cache="table-test", cache_dir=tmp_path)
    table_key = "metrics/table"
    artifact_name_contains = "table"
    cache._table_store(table_key, artifact_name_contains).save(
        project=cache.project,
        source_filters=None,
        table_key=table_key,
        artifact_name_contains=artifact_name_contains,
        include_summary=False,
        records=[
            {
                "run_id": "run-1",
                "run_name": "hopper-seed-1",
                "run_state": "finished",
                "run_group": None,
                "run_tags": ["pr-424"],
                "run_created_at": "2026-05-21T00:00:00Z",
                "config": {"env_id": "Hopper-v4", "seed": 1},
                "score": 100,
            }
        ],
    )

    df = cache.table_dataframe(
        table_key=table_key,
        artifact_name_contains=artifact_name_contains,
        config_keys=["env_id"],
    )

    assert df["score"].tolist() == [100]
    assert df["config.env_id"].tolist() == ["Hopper-v4"]
    assert "config.seed" not in df.columns


def cache_with_metadata(tmp_path: Path) -> WandbRunCache:
    cache = WandbRunCache(project="entity/project", cache="runs", cache_dir=tmp_path)

    def fetch_metadata_records(
        filters: dict[str, Any] | None,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
        graphql_per_page: int,
    ) -> list[dict[str, Any]]:
        return RUN_RECORDS

    cache._fetch_metadata_records = fetch_metadata_records  # type: ignore[method-assign]
    return cache
