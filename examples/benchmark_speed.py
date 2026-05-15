from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from wandb_cache import WandbRunCache


PROJECT = "ideas-ncbr/plan-crl"
EXP_NAME = "2026_05_13_self_refinement_generic_retry_13x4"
FILTERS = {"tags": EXP_NAME}
TABLE_KEY = "collect/episode_log"
ARTIFACT_NAME_CONTAINS = "episode_log"
PARALLEL_WORKERS = 32

T = TypeVar("T")


def timed(label: str, fn: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    value = fn()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
    return value, elapsed


def new_cache(name: str) -> WandbRunCache:
    return WandbRunCache(project=PROJECT, cache=f"benchmark/{EXP_NAME}/{name}")


def main() -> None:
    metadata_graphql, _ = timed(
        "metadata download + save, graphql",
        lambda: new_cache("metadata_graphql").refresh(filters=FILTERS, use_graphql=True),
    )
    print(f"metadata graphql runs: {len(metadata_graphql)}")

    metadata_wandb_api, _ = timed(
        "metadata download + save, wandb api",
        lambda: new_cache("metadata_wandb_api").refresh(filters=FILTERS, use_graphql=False),
    )
    print(f"metadata wandb api runs: {len(metadata_wandb_api)}")

    table_records_one_worker, _ = timed(
        "table refresh, 1 worker",
        lambda: new_cache("tables_1_worker").refresh_table(
            filters=FILTERS,
            table_key=TABLE_KEY,
            artifact_name_contains=ARTIFACT_NAME_CONTAINS,
            missing="skip",
            max_workers=1,
            use_graphql=True,
        ),
    )
    print(f"table rows, 1 worker: {len(table_records_one_worker)}")

    table_records_parallel, _ = timed(
        f"table refresh, {PARALLEL_WORKERS} workers",
        lambda: new_cache("tables_32_workers").refresh_table(
            filters=FILTERS,
            table_key=TABLE_KEY,
            artifact_name_contains=ARTIFACT_NAME_CONTAINS,
            missing="skip",
            max_workers=PARALLEL_WORKERS,
            use_graphql=True,
        ),
    )
    print(f"table rows, {PARALLEL_WORKERS} workers: {len(table_records_parallel)}")


if __name__ == "__main__":
    main()
