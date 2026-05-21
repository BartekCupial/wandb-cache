from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from wandb_cache.json import to_jsonable


def default_cache_path(cache_dir: str | Path, cache: str | Path | None, project: str) -> Path:
    if cache is None:
        cache = project

    cache_path = Path(cache)
    if cache_path.suffix:
        return cache_path

    safe_name = str(cache).strip("/").replace("/", "__")
    return Path(cache_dir) / f"{safe_name}.runs.parquet"


def run_metadata_cache_path(
    base_cache_path: str | Path,
    *,
    project: str,
    filters: dict[str, Any] | None,
    include_summary: bool,
    use_graphql: bool,
    graphql_filters: dict[str, Any] | None,
) -> Path:
    base_cache_path = Path(base_cache_path)
    base_name = base_cache_path.name.removesuffix(".runs.parquet")
    base_name = base_name.removesuffix(base_cache_path.suffix)
    spec = {
        "filters": to_jsonable(filters or {}),
        "graphql_filters": to_jsonable(graphql_filters or {}) if use_graphql else {},
        "include_summary": include_summary,
        "project": project,
        "use_graphql": use_graphql,
    }
    return base_cache_path.with_name(f"{base_name}.{_cache_digest(spec)}.runs.parquet")


def table_cache_path(
    run_cache_path: str | Path,
    table_key: str,
    artifact_name_contains: str,
    config_keys: Sequence[str] | None = None,
) -> Path:
    run_cache_path = Path(run_cache_path)
    base_name = run_cache_path.name
    base_name = base_name.removesuffix(".runs.parquet")
    safe_table_key = _safe_cache_token(table_key)
    safe_artifact = _safe_cache_token(artifact_name_contains)
    spec = {
        "artifact_name_contains": artifact_name_contains,
        "config_keys": list(config_keys or []),
        "table_key": table_key,
    }
    return run_cache_path.with_name(f"{base_name}.{safe_table_key}.{safe_artifact}.{_cache_digest(spec)}.table.parquet")


def history_cache_path(
    run_cache_path: str | Path,
    keys: Sequence[str] | None,
    samples: int,
    x_axis: str,
    stream: str,
    config_keys: Sequence[str] | None = None,
) -> Path:
    run_cache_path = Path(run_cache_path)
    base_name = run_cache_path.name
    base_name = base_name.removesuffix(".runs.parquet")
    spec = {
        "config_keys": list(config_keys or []),
        "keys": list(keys or []),
        "samples": samples,
        "stream": stream,
        "x_axis": x_axis,
    }
    safe_x_axis = _safe_cache_token(x_axis)
    return run_cache_path.with_name(f"{base_name}.{safe_x_axis}.{_cache_digest(spec)}.history.parquet")


def _safe_cache_token(value: str) -> str:
    return value.strip("/").replace("/", "__").replace(" ", "_")


def _cache_digest(value: dict[str, Any]) -> str:
    canonical = json.dumps(to_jsonable(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _encode_metadata(metadata: dict[str, Any]) -> dict[bytes, bytes]:
    encoded: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        encoded[key.encode("utf-8")] = json.dumps(value, ensure_ascii=True).encode("utf-8")
    return encoded


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    decoded: dict[str, Any] = {}
    for key, value in metadata.items():
        decoded[key.decode("utf-8")] = json.loads(value.decode("utf-8"))
    return decoded


def _records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    if not records:
        return pa.table({"_empty": pa.array([], type=pa.int8())})

    records = _drop_empty_dict_columns(records)
    keys = list(dict.fromkeys(key for record in records for key in record))
    return pa.Table.from_pylist([{key: record.get(key) for key in keys} for record in records])


def _drop_empty_dict_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {key for record in records for key in record}
    empty_dict_keys = {key for key in keys if all(record.get(key) == {} for record in records if key in record)}
    if not empty_dict_keys:
        return records

    cleaned_records = []
    for record in records:
        cleaned_record = dict(record)
        for key in empty_dict_keys:
            if cleaned_record.get(key) == {}:
                cleaned_record.pop(key)
        cleaned_records.append(cleaned_record)
    return cleaned_records


def _write_parquet(path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    table = _records_to_table(records).replace_schema_metadata(_encode_metadata(metadata))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(path)


class ParquetRunCacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        table = pq.read_table(self.path)
        payload = _decode_metadata(table.schema.metadata)
        payload["records"] = table.to_pylist()
        return payload

    def save(
        self,
        *,
        project: str,
        source_filters: dict[str, Any] | None,
        include_summary: bool,
        records: list[dict[str, Any]],
    ) -> None:
        payload = {
            "kind": "wandb_runs",
            "project": project,
            "source_filters": to_jsonable(source_filters or {}),
            "include_summary": include_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(records),
        }
        _write_parquet(self.path, records, payload)

    def exists(self) -> bool:
        return self.path.exists()


class ParquetTableCacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        table = pq.read_table(self.path)
        payload = _decode_metadata(table.schema.metadata)
        payload["records"] = table.to_pylist()
        return payload

    def save(
        self,
        *,
        project: str,
        source_filters: dict[str, Any] | None,
        table_key: str,
        artifact_name_contains: str,
        config_keys: Sequence[str] | None,
        include_summary: bool,
        records: list[dict[str, Any]],
    ) -> None:
        payload = {
            "kind": "wandb_table",
            "metadata_mode": "inline",
            "project": project,
            "source_filters": to_jsonable(source_filters or {}),
            "table_key": table_key,
            "artifact_name_contains": artifact_name_contains,
            "config_keys": to_jsonable(list(config_keys or [])),
            "include_summary": include_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(records),
        }
        _write_parquet(self.path, records, payload)

    def exists(self) -> bool:
        return self.path.exists()


class ParquetHistoryCacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        table = pq.read_table(self.path)
        payload = _decode_metadata(table.schema.metadata)
        payload["records"] = table.to_pylist()
        return payload

    def save(
        self,
        *,
        project: str,
        source_filters: dict[str, Any] | None,
        keys: Sequence[str] | None,
        samples: int,
        x_axis: str,
        stream: str,
        config_keys: Sequence[str] | None,
        include_summary: bool,
        records: list[dict[str, Any]],
    ) -> None:
        payload = {
            "kind": "wandb_history",
            "metadata_mode": "inline",
            "project": project,
            "source_filters": to_jsonable(source_filters or {}),
            "keys": to_jsonable(list(keys or [])),
            "samples": samples,
            "x_axis": x_axis,
            "stream": stream,
            "config_keys": to_jsonable(list(config_keys or [])),
            "include_summary": include_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(records),
        }
        _write_parquet(self.path, records, payload)

    def exists(self) -> bool:
        return self.path.exists()
