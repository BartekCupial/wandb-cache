from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wandb_cache import WandbRunCache

PROJECT = "openrlbenchmark/cleanrl"
EXP_NAME = "sac_continuous_action"
TAG = "pr-424"
FILTERS = {"$and": [{"tags": TAG}, {"config.exp_name": EXP_NAME}]}
GRAPHQL_FILTERS = {"tags": TAG}
CONFIG_KEYS = ["env_id", "exp_name", "seed"]
STEP_KEY = "global_step"
METRIC = "charts/episodic_return"
PLOT_ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
SAMPLES = 10_000
STEP_BIN_SIZE = 10_000
MAX_WORKERS = 8
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "figures" / "cleanrl_sac_history.png"


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache="public/cleanrl_sac")
    df = cache.history_dataframe(
        filters=FILTERS,
        refresh_cache=True,
        keys=[STEP_KEY, METRIC],
        samples=SAMPLES,
        x_axis=STEP_KEY,
        max_workers=MAX_WORKERS,
        config_keys=CONFIG_KEYS,
        use_graphql=True,
        graphql_filters=GRAPHQL_FILTERS,
    )

    if df.empty:
        raise RuntimeError(f"No history rows found for project {PROJECT!r}")

    df[STEP_KEY] = pd.to_numeric(df[STEP_KEY], errors="coerce")
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[STEP_KEY, METRIC, "config.env_id"])
    df = df[df["config.env_id"].isin(PLOT_ENVS)].copy()
    if df.empty:
        raise RuntimeError(f"No history rows found for envs: {PLOT_ENVS}")

    df["step_bin"] = (df[STEP_KEY] // STEP_BIN_SIZE) * STEP_BIN_SIZE
    per_run = (
        df.groupby(["config.env_id", "run_id", "step_bin"])[METRIC]
        .mean()
        .reset_index()
        .rename(columns={METRIC: "run_mean"})
    )
    stats = per_run.groupby(["config.env_id", "step_bin"])["run_mean"].agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"].fillna(0.0) / stats["count"].pow(0.5)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10").colors
    for index, env_id in enumerate(PLOT_ENVS):
        env_stats = stats[stats["config.env_id"] == env_id].sort_values("step_bin")
        if env_stats.empty:
            continue

        x = env_stats["step_bin"].to_numpy(dtype=float)
        mean = env_stats["mean"].to_numpy(dtype=float)
        sem = env_stats["sem"].to_numpy(dtype=float)
        color = colors[index % len(colors)]

        ax.plot(x, mean, label=env_id, color=color, linewidth=2.0)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)

    ax.set_title("CleanRL SAC episodic return")
    ax.set_xlabel("global step")
    ax.set_ylabel("episodic return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    print(f"Runs: {df['run_id'].nunique()}, rows: {len(df)}")


if __name__ == "__main__":
    main()
