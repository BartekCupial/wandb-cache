from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from wandb_cache.json import to_jsonable


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
    return _collect_worker_records(tasks, download_run_table_from_wandb, max_workers)


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


def get_run_table(run: Any, table_key: str, artifact_name_contains: str, missing: str) -> Any | None:
    if missing not in {"raise", "skip"}:
        raise ValueError("missing must be 'raise' or 'skip'")

    artifacts = [
        artifact
        for artifact in run.logged_artifacts()
        if artifact.type == "run_table" and artifact_name_contains in artifact.name and "describe" not in artifact.name
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


def serialize_table_rows(metadata: dict[str, Any], table: Any) -> list[dict[str, Any]]:
    conflicts = sorted(set(table.columns) & _metadata_column_names(metadata))
    if conflicts:
        raise ValueError(f"Table columns conflict with run metadata columns: {conflicts}")

    records = []
    for row_values in table.data:
        row = {column: to_jsonable(value) for column, value in zip(table.columns, row_values)}
        row.update(metadata)
        records.append(row)
    return records


def download_histories_from_metadata(
    project: str,
    metadata_records: list[dict[str, Any]],
    keys: Sequence[str] | None,
    samples: int,
    x_axis: str,
    stream: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    tasks = [
        {
            "project": project,
            "run_id": metadata["run_id"],
            "metadata": metadata,
            "keys": list(keys) if keys is not None else None,
            "samples": samples,
            "x_axis": x_axis,
            "stream": stream,
        }
        for metadata in metadata_records
    ]
    return _collect_worker_records(tasks, download_run_history_from_wandb, max_workers)


def download_run_history_from_wandb(task: dict[str, Any]) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api()
    run = api.run(f"{task['project']}/{task['run_id']}")
    history = run.history(
        keys=task["keys"],
        samples=task["samples"],
        x_axis=task["x_axis"],
        pandas=True,
        stream=task["stream"],
    )
    return serialize_history_rows(metadata=task["metadata"], history=history)


def serialize_history_rows(metadata: dict[str, Any], history: pd.DataFrame) -> list[dict[str, Any]]:
    if history.empty:
        return []

    conflicts = sorted(set(history.columns) & _metadata_column_names(metadata))
    if conflicts:
        raise ValueError(f"History columns conflict with run metadata columns: {conflicts}")

    records = []
    for row in history.where(pd.notna(history), None).to_dict(orient="records"):
        record = {column: to_jsonable(value) for column, value in row.items()}
        record.update(metadata)
        records.append(record)
    return records


def history_keys(keys: Sequence[str] | None, x_axis: str) -> list[str] | None:
    if keys is None:
        return None

    selected = list(dict.fromkeys(keys))
    if x_axis not in selected:
        selected.insert(0, x_axis)
    return selected


def _collect_worker_records(
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], list[dict[str, Any]]],
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_workers == 1:
        return _flatten_worker_batches(map(worker, tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return _flatten_worker_batches(executor.map(worker, tasks))


def _flatten_worker_batches(batches: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in batches:
        records.extend(batch)
    return records


def _metadata_column_names(metadata: dict[str, Any]) -> set[str]:
    columns = set(metadata)
    columns.add("config")
    return columns
