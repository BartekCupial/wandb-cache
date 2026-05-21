from pathlib import Path

from wandb_cache.cache import (
    ParquetRunCacheStore,
    ParquetTableCacheStore,
    default_cache_path,
    history_cache_path,
    run_metadata_cache_path,
    table_cache_path,
)


def test_default_cache_path_uses_project_when_cache_is_missing(tmp_path: Path) -> None:
    assert (
        default_cache_path(tmp_path, cache=None, project="entity/project") == tmp_path / "entity__project.runs.parquet"
    )


def test_default_cache_path_accepts_named_cache_and_explicit_file(tmp_path: Path) -> None:
    assert default_cache_path(tmp_path, cache="public/cleanrl_sac", project="ignored") == (
        tmp_path / "public__cleanrl_sac.runs.parquet"
    )
    assert (
        default_cache_path(tmp_path, cache=tmp_path / "custom.parquet", project="ignored")
        == tmp_path / "custom.parquet"
    )


def test_run_metadata_cache_path_depends_on_request() -> None:
    base_path = Path(".wandb_cache/public__cleanrl_sac.runs.parquet")

    filtered_path = run_metadata_cache_path(
        base_path,
        project="entity/project",
        filters={"tags": "pr-424"},
        include_summary=False,
        use_graphql=True,
        graphql_filters=None,
    )

    assert filtered_path == run_metadata_cache_path(
        base_path,
        project="entity/project",
        filters={"tags": "pr-424"},
        include_summary=False,
        use_graphql=True,
        graphql_filters=None,
    )
    assert filtered_path != run_metadata_cache_path(
        base_path,
        project="entity/project",
        filters={"tags": "other"},
        include_summary=False,
        use_graphql=True,
        graphql_filters=None,
    )
    assert filtered_path.name.startswith("public__cleanrl_sac.")
    assert filtered_path.name.endswith(".runs.parquet")


def test_table_and_history_cache_paths_are_stable() -> None:
    run_cache_path = Path(".wandb_cache/public__cleanrl_sac.runs.parquet")

    table_path = table_cache_path(
        run_cache_path,
        table_key="Table Name",
        artifact_name_contains="TableName",
        config_keys=["env_id"],
    )
    assert table_path == table_cache_path(
        run_cache_path,
        table_key="Table Name",
        artifact_name_contains="TableName",
        config_keys=["env_id"],
    )
    assert table_path != table_cache_path(
        run_cache_path,
        table_key="Table Name",
        artifact_name_contains="TableName",
        config_keys=["seed"],
    )
    assert table_path.name.startswith("public__cleanrl_sac.Table_Name.TableName.")
    assert table_path.name.endswith(".table.parquet")

    assert history_cache_path(
        run_cache_path,
        keys=["global_step", "charts/episodic_return"],
        samples=10_000,
        x_axis="global_step",
        stream="default",
        config_keys=["env_id"],
    ) == history_cache_path(
        run_cache_path,
        keys=["global_step", "charts/episodic_return"],
        samples=10_000,
        x_axis="global_step",
        stream="default",
        config_keys=["env_id"],
    )


def test_run_cache_roundtrip_drops_all_empty_dict_columns(tmp_path: Path) -> None:
    store = ParquetRunCacheStore(tmp_path / "runs.parquet")
    store.save(
        project="entity/project",
        source_filters={"tags": "example"},
        include_summary=False,
        records=[
            {
                "run_id": "run-1",
                "run_name": "empty-config",
                "run_state": "finished",
                "run_group": None,
                "run_tags": ["example"],
                "run_created_at": "2026-05-21T00:00:00Z",
                "config": {},
            }
        ],
    )

    payload = store.load()

    assert payload["project"] == "entity/project"
    assert payload["source_filters"] == {"tags": "example"}
    assert payload["row_count"] == 1
    assert payload["records"] == [
        {
            "run_id": "run-1",
            "run_name": "empty-config",
            "run_state": "finished",
            "run_group": None,
            "run_tags": ["example"],
            "run_created_at": "2026-05-21T00:00:00Z",
        }
    ]


def test_table_cache_roundtrip_preserves_union_of_record_columns(tmp_path: Path) -> None:
    store = ParquetTableCacheStore(tmp_path / "table.parquet")
    store.save(
        project="entity/project",
        source_filters={"tags": "mixed-envs"},
        table_key="collect/episode_log",
        artifact_name_contains="episode_log",
        config_keys=None,
        include_summary=False,
        records=[
            {"run_id": "code-run", "solved": 1.0},
            {"run_id": "arc-run", "task_success": 1.0},
        ],
    )

    payload = store.load()

    assert payload["records"] == [
        {"run_id": "code-run", "solved": 1.0, "task_success": None},
        {"run_id": "arc-run", "solved": None, "task_success": 1.0},
    ]
