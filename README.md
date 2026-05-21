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

Install pre-commit hooks for formatting and linting:

```bash
pre-commit install
```

## Metadata

```python
from wandb_cache import WandbRunCache

cache = WandbRunCache(
    project="openrlbenchmark/cleanrl",
    cache="cleanrl_sac",
)

df = cache.dataframe(
    filters={"$and": [{"tags": "pr-424"}, {"config.exp_name": "sac_continuous_action"}]},
    graphql_filters={"tags": "pr-424"},
    refresh_cache=True,
    config_keys=["env_id", "exp_name", "seed"],
    use_graphql=True,
)
```

The run metadata cache keeps the full W&B config. By default, config fields are not copied into returned
DataFrames or repeated table/history rows. This keeps large table and history Parquet files much smaller
because run config would otherwise be copied into every row. Pass `config_keys` to include only the config
fields you need:

```python
df = cache.dataframe(
    filters={"$and": [{"tags": "pr-424"}, {"config.exp_name": "sac_continuous_action"}]},
    graphql_filters={"tags": "pr-424"},
    refresh_cache=True,
    config_keys=["env_id", "seed"],
)
```

Leaving `config_keys` unset omits config columns from DataFrames and table/history row caches. Dotted keys
select nested config values, and a `config.` prefix is also accepted. When changing `config_keys` for an
existing table or history cache, use `refresh_cache=True` (or `refresh=True` with the function API) to
rewrite the Parquet file with the new config selection.

## Tables

```python
from wandb_cache import WandbRunCache

cache = WandbRunCache(
    project="carey/table-test",
    cache="table_test",
)

df = cache.table_dataframe(
    refresh_cache=True,
    table_key="Table Name",
    artifact_name_contains="TableName",
    missing="raise",
    max_workers=4,
    use_graphql=True,
)
```

## History

```python
from wandb_cache import WandbRunCache

cache = WandbRunCache(
    project="openrlbenchmark/cleanrl",
    cache="cleanrl_sac",
)

df = cache.history_dataframe(
    filters={"$and": [{"tags": "pr-424"}, {"config.exp_name": "sac_continuous_action"}]},
    graphql_filters={"tags": "pr-424"},
    refresh_cache=True,
    keys=[
        "global_step",
        "charts/episodic_return",
    ],
    samples=10_000,
    x_axis="global_step",
    max_workers=8,
    config_keys=["env_id", "exp_name", "seed"],
    use_graphql=True,
)
```

## Examples

```bash
python examples/metadata.py
python examples/history.py
python examples/table.py
python examples/benchmark.py
```

The metadata and history examples use the CleanRL SAC experiment from
`openrlbenchmark/cleanrl`, matching the [CleanRL SAC docs](https://docs.cleanrl.dev/rl-algorithms/sac/#experiment-results)
filter for `tag=pr-424` and `config.exp_name=sac_continuous_action`. The history example plots
`charts/episodic_return` with a standard-error shaded band across seeds.

The benchmark example uses the broader CleanRL `tag=pr-424` run set, which currently has enough public
runs to make the GraphQL speedup visible without requiring a private W&B project.

The table example uses the [W&B Tables walkthrough](https://docs.wandb.ai/models/tables/tables-walkthrough)
project `carey/table-test`. It is small, but useful as a stable smoke test for downloading run-table
artifacts into Parquet. All examples expect normal W&B auth through `WANDB_API_KEY` or `~/.netrc`.

## Benchmarking results

Command:

```bash
python examples/benchmark.py
```

Run on the public CleanRL `tag=pr-424` run set in `openrlbenchmark/cleanrl`: 198 runs.

**Metadata Benchmarks**
Comparing standard W&B API discovery versus GraphQL discovery, and network downloads versus local Parquet cache reads:

| Method | Network (refresh=True) | Cached (refresh=False) |
| :--- | ---: | ---: |
| **GraphQL** | 0.78s | 0.02s |
| **W&B API** | 30.19s | 0.01s |

## Roadmap

- [x] **History Tracking:** Add support for downloading, flattening, and caching sampled history metrics.
- [x] **Inline Metadata:** Inject run configs directly into Parquet rows for faster reads.
- [x] **Parquet Storage:** Migrate from JSON to Parquets, to speedup loading of the data.
- [x] **Benchmarking:** Add examples and benchmark the library.
- [x] **Multiprocessing:** Implement parallel workers to get through massive table downloads faster.
- [x] **Table Artifacts:** Add full support for downloading, flattening, and caching W&B tables.
- [x] **GraphQL:** Swap the standard W&B runs API for custom GraphQL to massively speed up metadata fetching.
