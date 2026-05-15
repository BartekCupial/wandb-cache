from __future__ import annotations

from time import perf_counter
from typing import Callable, Sized, TypeVar

from wandb_cache import WandbRunCache

PROJECT = "ideas-ncbr/plan-crl"
EXP_NAME = "2026_05_13_self_refinement_generic_retry_13x4"
FILTERS = {"tags": EXP_NAME}
TABLE_KEY = "collect/episode_log"
ARTIFACT_NAME_CONTAINS = "episode_log"
MAX_WORKERS = 32

T = TypeVar("T", bound=Sized)


def timed(label: str, fn: Callable[[], T]) -> None:
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    print(f"{label:<45} | {elapsed:>6.2f}s | {len(result)} rows")


def main() -> None:
    print("--- Metadata Benchmark ---")
    for graphql in (True, False):
        cache = WandbRunCache(project=PROJECT, cache=f"benchmark/metadata_gql_{graphql}")
        
        timed(
            f"Metadata (graphql={graphql}, refresh=True)",
            lambda: cache.dataframe(filters=FILTERS, use_graphql=graphql, refresh_cache=True)
        )
        timed(
            f"Metadata (graphql={graphql}, refresh=False)",
            lambda: cache.dataframe(filters=FILTERS, use_graphql=graphql, refresh_cache=False)
        )

    print("\n--- Table Benchmark ---")
    for graphql in (True, False):
        cache = WandbRunCache(project=PROJECT, cache=f"benchmark/table_gql_{graphql}")
        
        timed(
            f"Table (graphql={graphql}, refresh=True)",
            lambda: cache.table_dataframe(
                filters=FILTERS,
                table_key=TABLE_KEY,
                artifact_name_contains=ARTIFACT_NAME_CONTAINS,
                missing="skip",
                max_workers=MAX_WORKERS,
                use_graphql=graphql,
                refresh_cache=True,
            )
        )
        timed(
            f"Table (graphql={graphql}, refresh=False)",
            lambda: cache.table_dataframe(
                filters=FILTERS,
                table_key=TABLE_KEY,
                artifact_name_contains=ARTIFACT_NAME_CONTAINS,
                missing="skip",
                max_workers=MAX_WORKERS,
                use_graphql=graphql,
                refresh_cache=False,
            )
        )


if __name__ == "__main__":
    main()