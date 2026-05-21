from wandb_cache import WandbRunCache

PROJECT = "openrlbenchmark/cleanrl"
EXP_NAME = "sac_continuous_action"
TAG = "pr-424"
FILTERS = {"$and": [{"tags": TAG}, {"config.exp_name": EXP_NAME}]}
GRAPHQL_FILTERS = {"tags": TAG}
CONFIG_KEYS = ["env_id", "exp_name", "seed"]


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache="public/cleanrl_sac")
    df = cache.dataframe(
        filters=FILTERS,
        refresh_cache=True,
        config_keys=CONFIG_KEYS,
        use_graphql=True,
        graphql_filters=GRAPHQL_FILTERS,
    )

    print(f"Downloaded {len(df)} runs from {PROJECT}")
    print()
    table = (
        df.groupby(["config.env_id", "run_state"])
        .agg(runs=("run_id", "nunique"))
        .reset_index()
        .pivot_table(index="config.env_id", columns="run_state", values="runs", fill_value=0)
        .astype(int)
        .reset_index()
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
