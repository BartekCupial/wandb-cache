from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from wandb_cache.cache import (
    ParquetHistoryCacheStore,
    ParquetRunCacheStore,
    ParquetTableCacheStore,
    default_cache_path,
    history_cache_path,
    run_metadata_cache_path,
    table_cache_path,
)
from wandb_cache.configs import flatten_config_column, normalize_config_keys, select_record_configs
from wandb_cache.downloads import (
    download_histories_from_metadata,
    download_tables_from_metadata,
    history_keys,
    serialize_run_metadata,
)
from wandb_cache.filters import matches_filter
from wandb_cache.graphql import fetch_run_metadata_graphql


class WandbRunCache:
    def __init__(
        self,
        project: str,
        cache: str | Path | None = None,
        cache_dir: str | Path = ".wandb_cache",
    ):
        self.project = project
        self.cache_path = default_cache_path(cache_dir=cache_dir, cache=cache, project=project)
        self._api_client = None

    def dataframe(
        self,
        filters: dict[str, Any] | None = None,
        refresh_cache: bool = False,
        include_summary: bool = False,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
        config_keys: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        records = self._metadata_records(
            filters=filters,
            refresh_cache=refresh_cache,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        if not records:
            return pd.DataFrame()
        return pd.json_normalize(select_record_configs(records, config_keys), sep=".")

    def table_dataframe(
        self,
        filters: dict[str, Any] | None = None,
        refresh_cache: bool = False,
        table_key: str = "collect/episode_log",
        artifact_name_contains: str = "episode_log",
        include_summary: bool = False,
        missing: str = "raise",
        max_workers: int = 1,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
        config_keys: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        config_keys = normalize_config_keys(config_keys)
        store = self._table_store(
            filters=filters,
            table_key=table_key,
            artifact_name_contains=artifact_name_contains,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            config_keys=config_keys,
        )

        if refresh_cache or not store.exists():
            metadata = self._metadata_records(
                filters=filters,
                refresh_cache=refresh_cache,
                include_summary=include_summary,
                use_graphql=use_graphql,
                graphql_filters=graphql_filters,
                graphql_per_page=graphql_per_page,
            )
            records = download_tables_from_metadata(
                project=self.project,
                metadata_records=select_record_configs(metadata, config_keys),
                table_key=table_key,
                artifact_name_contains=artifact_name_contains,
                missing=missing,
                max_workers=max_workers,
            )
            store.save(
                project=self.project,
                source_filters=filters,
                table_key=table_key,
                artifact_name_contains=artifact_name_contains,
                config_keys=config_keys,
                include_summary=include_summary,
                records=records,
            )
        else:
            records = store.load()["records"]

        return flatten_config_column(pd.DataFrame.from_records(records), config_keys)

    def history_dataframe(
        self,
        filters: dict[str, Any] | None = None,
        refresh_cache: bool = False,
        keys: Sequence[str] | None = None,
        samples: int = 500,
        x_axis: str = "_step",
        stream: str = "default",
        include_summary: bool = False,
        max_workers: int = 1,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
        config_keys: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        config_keys = normalize_config_keys(config_keys)
        selected_history_keys = history_keys(keys=keys, x_axis=x_axis)
        store = self._history_store(
            filters=filters,
            keys=selected_history_keys,
            samples=samples,
            x_axis=x_axis,
            stream=stream,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            config_keys=config_keys,
        )

        if refresh_cache or not store.exists():
            metadata = self._metadata_records(
                filters=filters,
                refresh_cache=refresh_cache,
                include_summary=include_summary,
                use_graphql=use_graphql,
                graphql_filters=graphql_filters,
                graphql_per_page=graphql_per_page,
            )
            records = download_histories_from_metadata(
                project=self.project,
                metadata_records=select_record_configs(metadata, config_keys),
                keys=selected_history_keys,
                samples=samples,
                x_axis=x_axis,
                stream=stream,
                max_workers=max_workers,
            )
            store.save(
                project=self.project,
                source_filters=filters,
                keys=selected_history_keys,
                samples=samples,
                x_axis=x_axis,
                stream=stream,
                config_keys=config_keys,
                include_summary=include_summary,
                records=records,
            )
        else:
            records = store.load()["records"]

        if not records:
            return pd.DataFrame()
        return flatten_config_column(pd.DataFrame.from_records(records), config_keys)

    def _metadata_records(
        self,
        filters: dict[str, Any] | None,
        refresh_cache: bool,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
        graphql_per_page: int,
    ) -> list[dict[str, Any]]:
        store = self._metadata_store(
            filters=filters,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
        )
        if not refresh_cache and store.exists():
            return store.load()["records"]

        records = self._fetch_metadata_records(
            filters=filters,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        store.save(
            project=self.project,
            source_filters=filters,
            include_summary=include_summary,
            records=records,
        )
        return records

    def _fetch_metadata_records(
        self,
        filters: dict[str, Any] | None,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
        graphql_per_page: int,
    ) -> list[dict[str, Any]]:
        if use_graphql:
            records = fetch_run_metadata_graphql(
                project=self.project,
                filters=graphql_filters or filters,
                include_summary=include_summary,
                per_page=graphql_per_page,
            )
            return [record for record in records if matches_filter(record, filters)]

        return [
            serialize_run_metadata(run, include_summary=include_summary)
            for run in self._api().runs(self.project, filters=filters or {})
        ]

    def _metadata_cache_path(
        self,
        filters: dict[str, Any] | None,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
    ) -> Path:
        return run_metadata_cache_path(
            self.cache_path,
            project=self.project,
            filters=filters,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
        )

    def _metadata_store(
        self,
        filters: dict[str, Any] | None,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
    ) -> ParquetRunCacheStore:
        return ParquetRunCacheStore(
            self._metadata_cache_path(
                filters=filters,
                include_summary=include_summary,
                use_graphql=use_graphql,
                graphql_filters=graphql_filters,
            )
        )

    def _table_store(
        self,
        filters: dict[str, Any] | None,
        table_key: str,
        artifact_name_contains: str,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
        config_keys: Sequence[str] | None,
    ) -> ParquetTableCacheStore:
        return ParquetTableCacheStore(
            table_cache_path(
                run_cache_path=self._metadata_cache_path(
                    filters=filters,
                    include_summary=include_summary,
                    use_graphql=use_graphql,
                    graphql_filters=graphql_filters,
                ),
                table_key=table_key,
                artifact_name_contains=artifact_name_contains,
                config_keys=config_keys,
            )
        )

    def _history_store(
        self,
        filters: dict[str, Any] | None,
        keys: Sequence[str] | None,
        samples: int,
        x_axis: str,
        stream: str,
        include_summary: bool,
        use_graphql: bool,
        graphql_filters: dict[str, Any] | None,
        config_keys: Sequence[str] | None,
    ) -> ParquetHistoryCacheStore:
        return ParquetHistoryCacheStore(
            history_cache_path(
                run_cache_path=self._metadata_cache_path(
                    filters=filters,
                    include_summary=include_summary,
                    use_graphql=use_graphql,
                    graphql_filters=graphql_filters,
                ),
                keys=keys,
                samples=samples,
                x_axis=x_axis,
                stream=stream,
                config_keys=config_keys,
            )
        )

    def _api(self):
        if self._api_client is None:
            import wandb

            self._api_client = wandb.Api()
        return self._api_client
