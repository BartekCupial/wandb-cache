# wandb-cache

Small cache layer for W&B run metadata and table artifacts.

If you frequently pull W&B data to build pandas DataFrames for local analysis, plotting, or paper figures, you have likely hit bottlenecks with the slow `wandb` runs API. The standard `wandb.Api().runs(...)` call is often slow because it requests a massive data fragment for every run (including system metrics, history keys, and notes) and struggles with expensive server-side filtering on nested `config` fields.

`wandb-cache` solves this by fetching data through a custom, lightweight GraphQL query and caching it in fast Parquet files.

The intended workflow is:

1. Fetch a filtered set of runs from W&B.
2. Save raw-ish metadata and tables locally as Parquet.
3. Build pandas dataframes from the cache for plotting, summaries, and paper figures.

GraphQL is used by default for run metadata discovery because it is much faster for this use case. The normal W&B runs API is still available with `use_graphql=False`.

## Install

From the repository root:

```bash
pip install -e .
```

For packaging tools:

```bash
pip install -e ".[dev]"
```

## Metadata

```python
from wandb_cache import WandbRunCache

cache = WandbRunCache(
    project="ideas-ncbr/plan-crl",
    cache="self_refinement_generic_retry",
)

df = cache.dataframe(
    filters={"tags": "2026_05_13_self_refinement_generic_retry_13x4"},
    refresh_cache=True,
    use_graphql=True,
)
```

## Tables

```python
from wandb_cache import WandbRunCache

cache = WandbRunCache(
    project="ideas-ncbr/plan-crl",
    cache="self_refinement_generic_retry",
)

df = cache.table_dataframe(
    filters={"tags": "2026_05_13_self_refinement_generic_retry_13x4"},
    refresh_cache=True,
    table_key="collect/episode_log",
    artifact_name_contains="episode_log",
    missing="skip",
    max_workers=16,
    use_graphql=True,
)
```

## Examples

```bash
python examples/cache_metadata.py
python examples/cache_table_summary.py
python examples/benchmark_speed.py
```

These examples use the `ideas-ncbr/plan-crl` project and the `2026_05_13_self_refinement_generic_retry_13x4` tag, so they expect normal W&B auth through `WANDB_API_KEY` or `~/.netrc`.

## Benchmarking results

Command:

```bash
python examples/benchmark_speed.py
```

Run on `2026_05_13_self_refinement_generic_retry_13x4` in `ideas-ncbr/plan-crl`: 169 runs and 20,662 table rows.

**Metadata Benchmarks**
Comparing standard W&B API discovery versus GraphQL discovery, and network downloads versus local Parquet cache reads:

| Method | Network (refresh=True) | Cached (refresh=False) |
| :--- | ---: | ---: |
| **GraphQL** | 1.02s | 0.09s |
| **W&B API** | 45.06s | 0.12s |

**Table Benchmarks**
Table refresh timings include metadata selection, W&B table artifact downloads, table row serialization, and Parquet cache saves. Tests were run using 32 parallel workers (`max_workers=32`):

| Method | Network (refresh=True) | Cached (refresh=False) |
| :--- | ---: | ---: |
| **GraphQL** | 14.03s | 0.96s |
| **W&B API** | 57.32s | 1.13s |
