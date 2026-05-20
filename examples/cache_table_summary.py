import pandas as pd

from wandb_cache import WandbRunCache

PROJECT = "ideas-ncbr/plan-crl"
EXP_NAME = "2026_05_13_self_refinement_generic_retry_13x4"
FILTERS = {"tags": EXP_NAME}
TABLE_KEY = "collect/episode_log"
ARTIFACT_NAME_CONTAINS = "episode_log"


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache=f"examples/{EXP_NAME}")

    df = cache.table_dataframe(
        filters=FILTERS,
        refresh_cache=True,
        table_key=TABLE_KEY,
        artifact_name_contains=ARTIFACT_NAME_CONTAINS,
        missing="raise",
        max_workers=16,
        config_keys=["env_name", "llm_actor.engine_args.model_id"],
        use_graphql=True,
    )

    df["solved_at_1"] = (df["solved"] == 1.0) & (df["attempts_used"] <= 1)
    df["solved_at_10"] = (df["solved"] == 1.0) & (df["attempts_used"] <= 10)

    table = (
        df.groupby("config.env_name")
        .agg(
            runs=("run_id", "nunique"),
            episodes=("run_id", "size"),
            acc_at_1=("solved_at_1", "mean"),
            acc_at_10=("solved_at_10", "mean"),
        )
        .reset_index()
        .rename(columns={"config.env_name": "env"})
    )
    table["gain_at_10"] = table["acc_at_10"] - table["acc_at_1"]

    print(f"cached rows: {len(df)}")
    print()
    print(
        table.to_string(
            index=False,
            formatters={
                "acc_at_1": "{:.1%}".format,
                "acc_at_10": "{:.1%}".format,
                "gain_at_10": "+{:.1%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
