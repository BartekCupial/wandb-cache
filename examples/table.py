from wandb_cache import WandbRunCache

PROJECT = "carey/table-test"
TABLE_KEY = "Table Name"
ARTIFACT_NAME_CONTAINS = "TableName"


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache="public/table_test")
    df = cache.table_dataframe(
        refresh_cache=True,
        table_key=TABLE_KEY,
        artifact_name_contains=ARTIFACT_NAME_CONTAINS,
        missing="raise",
        max_workers=4,
        use_graphql=True,
    )

    print(f"Downloaded {len(df)} table rows from {PROJECT}")
    print()
    print(df[["run_id", "run_name", "a", "b"]].sort_values(["run_id", "a"]).to_string(index=False))


if __name__ == "__main__":
    main()
