#!/usr/bin/env python3
"""Plot and summarize the time distribution of four trajectory cut points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "output" / "cut_points.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "cut_point_distributions"

CUTS = {
    "cut1": {
        "column": "cut1_time",
        "title": "Cut 1: first mutation among all right-arm motors",
        "color": "#3977b8",
    },
    "cut2": {
        "column": "cut2_time",
        "title": "Cut 2: third right-gripper closure",
        "color": "#3d9970",
    },
    "cut3": {
        "column": "cut3_time",
        "title": "Cut 3: first mutation among all left-arm motors",
        "color": "#9b59b6",
    },
    "cut4": {
        "column": "cut4_time",
        "title": "Cut 4: second left-arm motor large change scanning backward",
        "color": "#d66a3a",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one time-distribution plot for each trajectory cut point."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def describe(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(values.count()),
        "mean_s": float(values.mean()),
        "std_s": float(values.std()),
        "min_s": float(values.min()),
        "p05_s": float(values.quantile(0.05)),
        "p25_s": float(values.quantile(0.25)),
        "median_s": float(values.median()),
        "p75_s": float(values.quantile(0.75)),
        "p95_s": float(values.quantile(0.95)),
        "max_s": float(values.max()),
    }


def plot_distribution(
    values: pd.Series, metadata: dict[str, str], stats: dict[str, float | int], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    bins = np.histogram_bin_edges(values.to_numpy(), bins="fd")
    axis.hist(
        values,
        bins=bins,
        color=metadata["color"],
        edgecolor="white",
        alpha=0.88,
    )
    axis.axvspan(
        stats["p05_s"], stats["p95_s"], color=metadata["color"], alpha=0.10,
        label="5th-95th percentile",
    )
    axis.axvline(
        stats["mean_s"], color="#c0392b", linewidth=2,
        label=f"Mean: {stats['mean_s']:.2f} s",
    )
    axis.axvline(
        stats["median_s"], color="#f39c12", linewidth=2, linestyle="--",
        label=f"Median: {stats['median_s']:.2f} s",
    )
    summary = (
        f"n = {stats['count']}\n"
        f"std = {stats['std_s']:.2f} s\n"
        f"P05 = {stats['p05_s']:.2f} s\n"
        f"P25 = {stats['p25_s']:.2f} s\n"
        f"P75 = {stats['p75_s']:.2f} s\n"
        f"P95 = {stats['p95_s']:.2f} s"
    )
    axis.text(
        0.98,
        0.96,
        summary,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set_title(metadata["title"])
    axis.set_xlabel("Time from episode start (seconds)")
    axis.set_ylabel("Number of episodes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "status" in frame.columns:
        frame = frame.loc[frame["status"] == "ok"].copy()
    missing = [item["column"] for item in CUTS.values() if item["column"] not in frame]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("No successful episodes found")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for cut_name, metadata in CUTS.items():
        values = pd.to_numeric(frame[metadata["column"]], errors="raise").dropna()
        stats = describe(values)
        summaries[cut_name] = {"description": metadata["title"], **stats}
        plot_distribution(
            values,
            metadata,
            stats,
            args.output / f"{cut_name}_time_distribution.png",
        )

    stats_frame = pd.DataFrame.from_dict(summaries, orient="index")
    stats_frame.index.name = "cut_point"
    stats_frame.to_csv(args.output / "cut_point_statistics.csv", float_format="%.6f")
    (args.output / "cut_point_statistics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated 4 plots and statistics for {len(frame)} episodes in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
