from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


def normalize_config_keys(config_keys: Sequence[str] | None) -> list[str]:
    if config_keys is None:
        return []

    keys: list[str] = []
    for key in [config_keys] if isinstance(config_keys, str) else config_keys:
        if not isinstance(key, str):
            raise TypeError("config_keys must contain strings")
        key = key.removeprefix("config.")
        if not key:
            raise ValueError("config_keys cannot contain empty strings")
        if key not in keys:
            keys.append(key)
    return keys


def select_record_configs(
    records: list[dict[str, Any]],
    config_keys: Sequence[str] | None,
) -> list[dict[str, Any]]:
    config_keys = normalize_config_keys(config_keys)

    selected_records = []
    for record in records:
        selected = dict(record)
        config = select_config_keys(record.get("config"), config_keys)
        if config:
            selected["config"] = config
        else:
            selected.pop("config", None)
        selected_records.append(selected)
    return selected_records


def select_config_keys(config: Any, config_keys: Sequence[str]) -> dict[str, Any]:
    source = dict(config or {}) if isinstance(config, dict) else {}
    selected: dict[str, Any] = {}
    for key in config_keys:
        if key in source:
            selected[key] = source[key]
            continue

        found, value = _read_nested_config(source, key.split("."))
        if found:
            _write_nested_config(selected, key.split("."), value)
    return selected


def flatten_config_column(df: pd.DataFrame, config_keys: Sequence[str] | None = None) -> pd.DataFrame:
    if "config" not in df.columns:
        return df

    config_keys = normalize_config_keys(config_keys)
    if not config_keys:
        return df.drop(columns=["config"])

    configs = [select_config_keys(config, config_keys) for config in df["config"].tolist()]
    config_df = pd.json_normalize(configs, sep=".").add_prefix("config.")
    config_df.index = df.index
    return pd.concat([df.drop(columns=["config"]), config_df], axis=1)


def _read_nested_config(config: dict[str, Any], parts: Sequence[str]) -> tuple[bool, Any]:
    value: Any = config
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _write_nested_config(config: dict[str, Any], parts: Sequence[str], value: Any) -> None:
    target = config
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = {}
            target[part] = nested
        target = nested
    target[parts[-1]] = value
