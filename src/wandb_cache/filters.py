from __future__ import annotations

import re
from typing import Any

FIELD_ALIASES = {
    "id": "run_id",
    "name": "run_id",
    "display_name": "run_name",
    "state": "run_state",
    "group": "run_group",
    "tags": "run_tags",
}


def matches_filter(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    for key, condition in filters.items():
        if key == "$or":
            if not any(matches_filter(record, item) for item in condition):
                return False
            continue
        if key == "$and":
            if not all(matches_filter(record, item) for item in condition):
                return False
            continue

        try:
            value = read_field(record, key)
        except KeyError:
            return False
        if not matches_condition(value, condition):
            return False
    return True


def read_field(record: dict[str, Any], field: str) -> Any:
    field = FIELD_ALIASES.get(field, field)
    parts = field.split(".")
    value: Any = record
    for part in parts:
        value = value[part]
    return value


def matches_condition(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict) and any(str(key).startswith("$") for key in condition):
        for operator, expected in condition.items():
            if not matches_operator(value, operator, expected):
                return False
        return True

    return matches_equality(value, condition)


def matches_operator(value: Any, operator: str, expected: Any) -> bool:
    if operator == "$eq":
        return matches_equality(value, expected)
    if operator == "$ne":
        return not matches_equality(value, expected)
    if operator == "$in":
        return matches_in(value, expected)
    if operator == "$nin":
        return not matches_in(value, expected)
    if operator == "$regex":
        return re.search(str(expected), str(value)) is not None
    if operator == "$gt":
        return value > expected
    if operator == "$gte":
        return value >= expected
    if operator == "$lt":
        return value < expected
    if operator == "$lte":
        return value <= expected
    raise ValueError(f"Unsupported filter operator: {operator}")


def matches_equality(value: Any, expected: Any) -> bool:
    if isinstance(value, list):
        if isinstance(expected, list):
            return all(item in value for item in expected)
        return expected in value
    return value == expected


def matches_in(value: Any, expected: Any) -> bool:
    if not isinstance(expected, list | tuple | set):
        raise TypeError("$in and $nin expect a list, tuple, or set")

    if isinstance(value, list):
        return any(item in expected for item in value)
    return value in expected
