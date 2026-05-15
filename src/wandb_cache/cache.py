from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wandb_cache.json import to_jsonable


def default_cache_path(cache_dir: str | Path, cache: str | Path | None, project: str) -> Path:
    if cache is None:
        cache = project

    cache_path = Path(cache)
    if cache_path.suffix:
        return cache_path

    safe_name = str(cache).strip("/").replace("/", "__")
    return Path(cache_dir) / f"{safe_name}.runs.json"


def table_cache_path(run_cache_path: str | Path, table_key: str, artifact_name_contains: str) -> Path:
    run_cache_path = Path(run_cache_path)
    base_name = run_cache_path.name
    base_name = base_name.removesuffix(".runs.json")
    safe_table_key = _safe_cache_token(table_key)
    safe_artifact = _safe_cache_token(artifact_name_contains)
    return run_cache_path.with_name(f"{base_name}.{safe_table_key}.{safe_artifact}.table.json")


def _safe_cache_token(value: str) -> str:
    return value.strip("/").replace("/", "__").replace(" ", "_")


class JsonRunCacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

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
            "records": records,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    def exists(self) -> bool:
        return self.path.exists()


class JsonTableCacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(
        self,
        *,
        project: str,
        source_filters: dict[str, Any] | None,
        table_key: str,
        artifact_name_contains: str,
        include_summary: bool,
        records: list[dict[str, Any]],
    ) -> None:
        payload = {
            "kind": "wandb_table",
            "metadata_mode": "separate",
            "project": project,
            "source_filters": to_jsonable(source_filters or {}),
            "table_key": table_key,
            "artifact_name_contains": artifact_name_contains,
            "include_summary": include_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(records),
            "records": records,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    def exists(self) -> bool:
        return self.path.exists()
