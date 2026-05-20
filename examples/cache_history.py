from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wandb_cache import WandbRunCache

PROJECT = "jiayipan/TinyZero"
CACHE_NAME = "tinyzero_core"
FILTERS = None
STEP_KEY = "_step"
METRICS = [
    "critic/score/mean",
    "response_length/mean",
    "timing_s/step",
]
SAMPLES = 10_000
MAX_WORKERS = 16
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "figures" / "tinyzero_core_history.png"


def main() -> None:
    cache = WandbRunCache(project=PROJECT, cache=f"examples/{CACHE_NAME}")
    df = cache.history_dataframe(
        filters=FILTERS,
        refresh_cache=True,
        keys=[STEP_KEY, *METRICS],
        samples=SAMPLES,
        x_axis=STEP_KEY,
        max_workers=MAX_WORKERS,
        use_graphql=True,
    )

    if df.empty:
        raise RuntimeError(f"No history rows found for project {PROJECT!r}")

    for column in [STEP_KEY, *METRICS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[STEP_KEY]).sort_values(["run_id", STEP_KEY])

    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5), sharex=True)
    fig.suptitle(f"{PROJECT} core metrics")

    handles = []
    labels = []
    colors = plt.get_cmap("tab20").colors
    run_groups = list(df.groupby("run_id", sort=False))
    for ax, metric in zip(axes, METRICS):
        metric_df = df[[STEP_KEY, metric, "run_id", "run_name"]].dropna()
        for index, (run_id, _) in enumerate(run_groups):
            run_metric_df = metric_df[metric_df["run_id"] == run_id]
            if run_metric_df.empty:
                continue

            label = f"{run_metric_df['run_name'].iloc[0]} ({run_id})"
            (line,) = ax.plot(
                run_metric_df[STEP_KEY],
                run_metric_df[metric],
                color=colors[index % len(colors)],
                linewidth=1.4,
                alpha=0.9,
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(label)

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xlabel("step")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.77, 0.9))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    print(f"Runs: {df['run_id'].nunique()}, rows: {len(df)}")


if __name__ == "__main__":
    main()
