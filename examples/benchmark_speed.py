from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from wandb_cache import WandbRunCache


PROJECT = "ideas-ncbr/plan-crl"
EXP_NAME = "2026_05_13_self_refinement_generic_retry_13x4"
FILTERS = {"tags": EXP_NAME}
TABLE_KEY = "collect/episode_log"
ARTIFACT_NAME_CONTAINS = "episode_log"
MAX_WORKERS = 16

T = TypeVar("T")


def timed(label: str, fn: Callable[[], T]) -> T:
    start = perf_counter()
    value = fn()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
    return value


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache=f"examples/{EXP_NAME}")

    metadata = timed(
        "metadata download + save",
        lambda: cache.refresh(filters=FILTERS, use_graphql=True),
    )
    print(f"metadata runs: {len(metadata)}")

    table_records = timed(
        "table refresh (metadata + tables + save)",
        lambda: cache.refresh_table(
            filters=FILTERS,
            table_key=TABLE_KEY,
            artifact_name_contains=ARTIFACT_NAME_CONTAINS,
            missing="skip",
            max_workers=MAX_WORKERS,
            use_graphql=True,
        ),
    )
    print(f"table rows: {len(table_records)}")

    df = timed(
        "cached table dataframe",
        lambda: cache.table_dataframe(
            filters=FILTERS,
            table_key=TABLE_KEY,
            artifact_name_contains=ARTIFACT_NAME_CONTAINS,
            refresh_cache=False,
        ),
    )
    print(f"dataframe shape: {df.shape}")


if __name__ == "__main__":
    main()
