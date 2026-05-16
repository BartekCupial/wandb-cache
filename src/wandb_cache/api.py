from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from wandb_cache.runs import WandbRunCache


def fetch_runs(
    project: str,
    filters: dict[str, Any] | None = None,
    cache: str | Path | None = None,
    cache_dir: str | Path = ".wandb_cache",
    refresh: bool = False,
    include_summary: bool = False,
    use_graphql: bool = True,
    graphql_filters: dict[str, Any] | None = None,
    graphql_per_page: int = 500,
) -> pd.DataFrame:
    run_cache = WandbRunCache(project=project, cache=cache, cache_dir=cache_dir)
    return run_cache.dataframe(
        filters=filters,
        refresh_cache=refresh,
        include_summary=include_summary,
        use_graphql=use_graphql,
        graphql_filters=graphql_filters,
        graphql_per_page=graphql_per_page,
    )


def fetch_table(
    project: str,
    filters: dict[str, Any] | None = None,
    table_key: str = "collect/episode_log",
    artifact_name_contains: str = "episode_log",
    cache: str | Path | None = None,
    cache_dir: str | Path = ".wandb_cache",
    refresh: bool = False,
    include_summary: bool = False,
    missing: str = "raise",
    max_workers: int = 1,
    use_graphql: bool = True,
    graphql_filters: dict[str, Any] | None = None,
    graphql_per_page: int = 500,
) -> pd.DataFrame:
    run_cache = WandbRunCache(project=project, cache=cache, cache_dir=cache_dir)
    return run_cache.table_dataframe(
        filters=filters,
        refresh_cache=refresh,
        table_key=table_key,
        artifact_name_contains=artifact_name_contains,
        include_summary=include_summary,
        missing=missing,
        max_workers=max_workers,
        use_graphql=use_graphql,
        graphql_filters=graphql_filters,
        graphql_per_page=graphql_per_page,
    )


def fetch_history(
    project: str,
    filters: dict[str, Any] | None = None,
    keys: Sequence[str] | None = None,
    samples: int = 500,
    x_axis: str = "_step",
    stream: str = "default",
    cache: str | Path | None = None,
    cache_dir: str | Path = ".wandb_cache",
    refresh: bool = False,
    include_summary: bool = False,
    max_workers: int = 1,
    use_graphql: bool = True,
    graphql_filters: dict[str, Any] | None = None,
    graphql_per_page: int = 500,
) -> pd.DataFrame:
    run_cache = WandbRunCache(project=project, cache=cache, cache_dir=cache_dir)
    return run_cache.history_dataframe(
        filters=filters,
        refresh_cache=refresh,
        keys=keys,
        samples=samples,
        x_axis=x_axis,
        stream=stream,
        include_summary=include_summary,
        max_workers=max_workers,
        use_graphql=use_graphql,
        graphql_filters=graphql_filters,
        graphql_per_page=graphql_per_page,
    )
