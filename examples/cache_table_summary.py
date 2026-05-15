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
        missing="skip",
        max_workers=16,
        use_graphql=True,
    )

    summary = pd.DataFrame(
        [
            {
                "runs": df["run_id"].nunique(),
                "rows": len(df),
                "solved_at_1": df["solved_at_1"].mean(),
                "solved_at_10": df["solved_at_10"].mean(),
            }
        ]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
