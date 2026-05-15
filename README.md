# wandb-cache

Small cache layer for W&B run metadata and table artifacts.

The intended workflow is:

1. Fetch a filtered set of runs from W&B.
2. Save raw-ish metadata and tables locally as JSON.
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

The table cache stores table rows separately from run metadata. Metadata is attached when building the dataframe, which keeps table saves much smaller and faster.

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

Metadata download and JSON cache save:

| Variant | Time | Speedup |
| --- | ---: | ---: |
| GraphQL metadata | 1.35s | 19.6x |
| W&B runs API metadata | 26.45s | 1.0x |

Table refresh, JSON cache save included:

| Variant | Time | Speedup |
| --- | ---: | ---: |
| 1 worker | 313.99s | 1.0x |
| 32 workers | 12.09s | 26.0x |

The table refresh timing includes GraphQL metadata selection, W&B table artifact download, table row serialization, and JSON cache save. Network conditions and the local W&B artifact cache can move these numbers around, but the benchmark gives the right scale: GraphQL helps metadata discovery, and parallel workers are the big win for table artifacts.

