# wandb-cache

[![CI](https://github.com/BartekCupial/wandb-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/BartekCupial/wandb-cache/actions/workflows/ci.yml)

Fast local caching for W&B run metadata, history metrics, and table artifacts.

`wandb-cache` is for the very common research workflow where you pull W&B data into pandas for
analysis, plotting, sweeps, or paper figures. The standard `wandb.Api().runs(...)` path can be slow because
it requests a large GraphQL fragment for every run and can struggle with expensive server-side filtering on
nested `config` fields.

This library uses a small custom GraphQL query for run discovery, filters locally when that is faster, and
caches the result as Parquet. Once cached, rebuilding a DataFrame is usually a local disk read.

```text
openrlbenchmark/cleanrl, tag=pr-424, 198 runs

GraphQL refresh: 0.78s
W&B API refresh: 30.19s
Cached read:     0.02s
```

## Install

From PyPI, once released:

```bash
pip install wandb-cache
```

From the repository root:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Quick Start

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
)

print(df[["run_id", "run_name", "config.env_id", "config.seed"]].head())
```

Normal W&B auth is used through `WANDB_API_KEY` or `~/.netrc`.

## Config Columns

The run metadata cache keeps the full W&B config. By default, config fields are not copied into returned
DataFrames or repeated table/history rows. This keeps large table and history Parquet files much smaller
because run config would otherwise be copied into every row.

Pass `config_keys` to include only the config fields you need:

```python
df = cache.dataframe(
    filters={"tags": "pr-424"},
    refresh_cache=True,
    config_keys=["env_id", "seed", "llm_actor.engine_args.model_id"],
)
```

Dotted keys select nested config values, and a `config.` prefix is also accepted.

Cache filenames include a deterministic hash of the request that created them, including filters, GraphQL
filters, summary inclusion, and table/history config selections. Changing those inputs creates a separate
Parquet file instead of reinterpreting an existing cache. Use `refresh_cache=True` when you want to overwrite
the cache for the same request and pick up new W&B data.

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
artifacts into Parquet.

## Benchmark

Command:

```bash
python examples/benchmark.py
```

Run on the public CleanRL `tag=pr-424` run set in `openrlbenchmark/cleanrl`: 198 runs.

| Method | Network Refresh | Cached Read |
| :--- | ---: | ---: |
| GraphQL | 0.78s | 0.02s |
| W&B API | 30.19s | 0.01s |

## Release Status

`wandb-cache` is early and intentionally small. The current API is useful for metadata, sampled history,
and table artifacts, but may still change while the first public users kick the tires.
