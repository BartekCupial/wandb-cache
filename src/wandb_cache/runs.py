from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from wandb_cache.cache import JsonRunCacheStore, JsonTableCacheStore, default_cache_path, table_cache_path
from wandb_cache.filters import matches_filter
from wandb_cache.graphql import fetch_run_metadata_graphql
from wandb_cache.json import to_jsonable


class WandbRunCache:
    def __init__(
        self,
        project: str,
        cache: str | Path | None = None,
        cache_dir: str | Path = ".wandb_cache",
    ):
        self.project = project
        self.cache_path = default_cache_path(cache_dir=cache_dir, cache=cache, project=project)
        self.store = JsonRunCacheStore(self.cache_path)
        self._api_client = None

    def _fetch_runs(self, filters: dict[str, Any] | None = None) -> list[Any]:
        api = self._api()
        return list(api.runs(self.project, filters=filters or {}))

    def refresh(
        self,
        filters: dict[str, Any] | None = None,
        include_summary: bool = False,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
    ) -> list[dict[str, Any]]:
        records = self._fetch_metadata_records(
            filters=filters,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        self._save_metadata(filters=filters, include_summary=include_summary, records=records)
        return records

    def _save_metadata(
        self,
        filters: dict[str, Any] | None,
        include_summary: bool,
        records: list[dict[str, Any]],
    ) -> None:
        self.store.save(
            project=self.project,
            source_filters=filters,
            include_summary=include_summary,
            records=records,
        )

    def records(
        self,
        filters: dict[str, Any] | None = None,
        refresh_cache: bool = False,
        include_summary: bool = False,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
    ) -> list[dict[str, Any]]:
        if refresh_cache:
            records = self.refresh(
                filters=filters,
                include_summary=include_summary,
                use_graphql=use_graphql,
                graphql_filters=graphql_filters,
                graphql_per_page=graphql_per_page,
            )
        else:
            if not self.store.exists():
                records = self.refresh(
                    filters=filters,
                    include_summary=include_summary,
                    use_graphql=use_graphql,
                    graphql_filters=graphql_filters,
                    graphql_per_page=graphql_per_page,
                )
            else:
                payload = self.store.load()
                cached_include_summary = payload.get("include_summary", False)
                if cached_include_summary != include_summary:
                    raise ValueError(
                        "Cached run metadata was saved with "
                        f"include_summary={cached_include_summary}; requested include_summary={include_summary}. "
                        "Refresh the cache to change this."
                    )
                records = payload["records"]

        return [record for record in records if matches_filter(record, filters)]

    def dataframe(
        self,
        filters: dict[str, Any] | None = None,
        refresh_cache: bool = False,
        include_summary: bool = False,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
    ) -> pd.DataFrame:
        records = self.records(
            filters=filters,
            refresh_cache=refresh_cache,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        if not records:
            return pd.DataFrame()
        return pd.json_normalize(records, sep=".")

    def refresh_table(
        self,
        filters: dict[str, Any] | None = None,
        table_key: str = "collect/episode_log",
        artifact_name_contains: str = "episode_log",
        include_summary: bool = False,
        missing: str = "raise",
        max_workers: int = 1,
        use_graphql: bool = True,
        graphql_filters: dict[str, Any] | None = None,
        graphql_per_page: int = 500,
    ) -> list[dict[str, Any]]:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        metadata_records = self._fetch_metadata_records(
            filters=filters,
            include_summary=include_summary,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        self._save_metadata(filters=filters, include_summary=include_summary, records=metadata_records)

        table_records = download_tables_from_metadata(
            project=self.project,
            metadata_records=metadata_records,
            table_key=table_key,
            artifact_name_contains=artifact_name_contains,
            missing=missing,
            max_workers=max_workers,
        )

        store = self._table_store(table_key=table_key, artifact_name_contains=artifact_name_contains)
        store.save(
            project=self.project,
            source_filters=filters,
            table_key=table_key,
            artifact_name_contains=artifact_name_contains,
            include_summary=include_summary,
            records=table_records,
        )
        return attach_metadata_to_table_records(table_records, metadata_records)

    def table_records(
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
    ) -> list[dict[str, Any]]:
        store = self._table_store(table_key=table_key, artifact_name_contains=artifact_name_contains)
        if refresh_cache or not store.exists():
            records = self.refresh_table(
                filters=filters,
                table_key=table_key,
                artifact_name_contains=artifact_name_contains,
                include_summary=include_summary,
                missing=missing,
                max_workers=max_workers,
                use_graphql=use_graphql,
                graphql_filters=graphql_filters,
                graphql_per_page=graphql_per_page,
            )
        else:
            payload = store.load()
            cached_include_summary = payload.get("include_summary", False)
            if cached_include_summary != include_summary:
                raise ValueError(
                    "Cached table data was saved with "
                    f"include_summary={cached_include_summary}; requested include_summary={include_summary}. "
                    "Refresh the cache to change this."
                )
            records = payload["records"]
            metadata_records = self._load_metadata(include_summary=include_summary)
            records = attach_metadata_to_table_records(records, metadata_records)

        return [record for record in records if matches_filter(record, filters)]

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
    ) -> pd.DataFrame:
        records = self.table_records(
            filters=filters,
            refresh_cache=refresh_cache,
            table_key=table_key,
            artifact_name_contains=artifact_name_contains,
            include_summary=include_summary,
            missing=missing,
            max_workers=max_workers,
            use_graphql=use_graphql,
            graphql_filters=graphql_filters,
            graphql_per_page=graphql_per_page,
        )
        if not records:
            return pd.DataFrame()
        return pd.json_normalize(records, sep=".")

    def refresh_history(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("History caching is not implemented yet.")

    def _table_store(self, table_key: str, artifact_name_contains: str) -> JsonTableCacheStore:
        return JsonTableCacheStore(
            table_cache_path(
                run_cache_path=self.cache_path,
                table_key=table_key,
                artifact_name_contains=artifact_name_contains,
            )
        )

    def _load_metadata(self, include_summary: bool) -> list[dict[str, Any]]:
        if not self.store.exists():
            raise ValueError("Table cache stores metadata separately, but the run metadata cache is missing.")

        payload = self.store.load()
        cached_include_summary = payload.get("include_summary", False)
        if cached_include_summary != include_summary:
            raise ValueError(
                "Cached run metadata was saved with "
                f"include_summary={cached_include_summary}; requested include_summary={include_summary}. "
                "Refresh the cache to change this."
            )
        return payload["records"]

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

        runs = self._fetch_runs(filters=filters)
        return [serialize_run_metadata(run, include_summary=include_summary) for run in runs]

    def _api(self):
        if self._api_client is not None:
            return self._api_client

        import wandb

        self._api_client = wandb.Api()
        return self._api_client


def serialize_run_metadata(run: Any, include_summary: bool = False) -> dict[str, Any]:
    record = {
        "run_id": to_jsonable(run.id),
        "run_name": to_jsonable(run.name),
        "run_state": to_jsonable(run.state),
        "run_group": to_jsonable(run.group),
        "run_tags": to_jsonable(list(run.tags)),
        "run_created_at": to_jsonable(run.created_at),
        "config": to_jsonable(dict(run.config)),
    }
    if include_summary:
        record["summary"] = to_jsonable(dict(run.summary))
    return record


def get_run_table(
    run: Any,
    table_key: str,
    artifact_name_contains: str,
    missing: str,
) -> Any | None:
    if missing not in {"raise", "skip"}:
        raise ValueError("missing must be 'raise' or 'skip'")

    artifacts = [
        artifact
        for artifact in run.logged_artifacts()
        if artifact.type == "run_table"
        and artifact_name_contains in artifact.name
        and "describe" not in artifact.name
    ]
    if not artifacts:
        if missing == "skip":
            return None
        raise ValueError(f"No table artifact matching {artifact_name_contains!r} found for run {run.id}")

    table = artifacts[-1].get(table_key)
    if table is None:
        if missing == "skip":
            return None
        raise ValueError(f"No table key {table_key!r} found in artifact {artifacts[-1].name!r} for run {run.id}")
    return table


def download_tables_from_metadata(
    project: str,
    metadata_records: list[dict[str, Any]],
    table_key: str,
    artifact_name_contains: str,
    missing: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    tasks = [
        {
            "project": project,
            "run_id": metadata["run_id"],
            "metadata": metadata,
            "table_key": table_key,
            "artifact_name_contains": artifact_name_contains,
            "missing": missing,
        }
        for metadata in metadata_records
    ]

    table_records: list[dict[str, Any]] = []
    if max_workers == 1:
        for task in tasks:
            table_records.extend(download_run_table_from_wandb(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for run_records in executor.map(download_run_table_from_wandb, tasks):
                table_records.extend(run_records)
    return table_records


def download_run_table_from_wandb(task: dict[str, Any]) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api()
    run = api.run(f"{task['project']}/{task['run_id']}")
    table = get_run_table(
        run,
        table_key=task["table_key"],
        artifact_name_contains=task["artifact_name_contains"],
        missing=task["missing"],
    )
    if table is None:
        return []
    return serialize_table_rows(metadata=task["metadata"], table=table)


def serialize_table_rows(metadata: dict[str, Any], table: Any) -> list[dict[str, Any]]:
    records = []
    for row_values in table.data:
        row = {column: to_jsonable(value) for column, value in zip(table.columns, row_values)}
        conflicting_columns = sorted(set(row) & {"run_id"})
        if conflicting_columns:
            raise ValueError(f"Table columns conflict with run metadata columns: {conflicting_columns}")
        row["run_id"] = metadata["run_id"]
        records.append(row)
    return records


def attach_metadata_to_table_records(
    table_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_run_id = {metadata["run_id"]: metadata for metadata in metadata_records}
    records = []
    for table_record in table_records:
        metadata = metadata_by_run_id[table_record["run_id"]]
        conflicts = sorted((set(table_record) & set(metadata)) - {"run_id"})
        if conflicts:
            raise ValueError(f"Table columns conflict with run metadata columns: {conflicts}")
        record = dict(table_record)
        record.update(metadata)
        records.append(record)
    return records
