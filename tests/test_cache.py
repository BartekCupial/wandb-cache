from pathlib import Path

from wandb_cache.cache import (
    ParquetRunCacheStore,
    default_cache_path,
    history_cache_path,
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


def test_table_and_history_cache_paths_are_stable() -> None:
    run_cache_path = Path(".wandb_cache/public__cleanrl_sac.runs.parquet")

    assert table_cache_path(run_cache_path, table_key="Table Name", artifact_name_contains="TableName") == Path(
        ".wandb_cache/public__cleanrl_sac.Table_Name.TableName.table.parquet"
    )
    assert history_cache_path(
        run_cache_path,
        keys=["global_step", "charts/episodic_return"],
        samples=10_000,
        x_axis="global_step",
        stream="default",
    ) == history_cache_path(
        run_cache_path,
        keys=["global_step", "charts/episodic_return"],
        samples=10_000,
        x_axis="global_step",
        stream="default",
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
