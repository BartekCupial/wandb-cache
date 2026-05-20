from wandb_cache import WandbRunCache


PROJECT = "ideas-ncbr/plan-crl"
EXP_NAME = "2026_05_13_self_refinement_generic_retry_13x4"
FILTERS = {"tags": EXP_NAME}


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache=f"examples/{EXP_NAME}")

    df = cache.dataframe(
        filters=FILTERS,
        refresh_cache=True,
        use_graphql=True,
    )

    print(f"Downloaded {len(df)} runs")
    print(df[["run_id", "run_name", "run_state", "run_created_at"]].head())


if __name__ == "__main__":
    main()
