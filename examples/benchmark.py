from __future__ import annotations

from time import perf_counter
from typing import Callable, Sized, TypeVar

from wandb_cache import WandbRunCache

PROJECT = "openrlbenchmark/cleanrl"
TAG = "pr-424"
FILTERS = {"tags": TAG}
GRAPHQL_FILTERS = {"tags": TAG}
CONFIG_KEYS = ["env_id", "exp_name", "seed"]

T = TypeVar("T", bound=Sized)


def timed(label: str, fn: Callable[[], T]) -> None:
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    print(f"{label:<45} | {elapsed:>6.2f}s | {len(result)} rows")


def dataframe(cache: WandbRunCache, use_graphql: bool, refresh_cache: bool):
    return cache.dataframe(
        filters=FILTERS,
        refresh_cache=refresh_cache,
        config_keys=CONFIG_KEYS,
        use_graphql=use_graphql,
        graphql_filters=GRAPHQL_FILTERS if use_graphql else None,
    )


def main() -> None:
    print("--- Public Metadata Benchmark ---")
    print(f"Project: {PROJECT}")
    print(f"Runs: tag={TAG!r}")
    print()

    for use_graphql in (True, False):
        cache = WandbRunCache(project=PROJECT, cache=f"public/benchmark_cleanrl_gql_{use_graphql}")
        timed(
            f"Metadata (graphql={use_graphql}, refresh=True)",
            lambda use_graphql=use_graphql, cache=cache: dataframe(
                cache,
                use_graphql=use_graphql,
                refresh_cache=True,
            ),
        )
        timed(
            f"Metadata (graphql={use_graphql}, refresh=False)",
            lambda use_graphql=use_graphql, cache=cache: dataframe(
                cache,
                use_graphql=use_graphql,
                refresh_cache=False,
            ),
        )


if __name__ == "__main__":
    main()
