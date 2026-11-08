"""Reusable exploratory data analysis summaries and plots."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.config import NUMERIC_FEATURES, REPORTS_DIR, TARGET

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when the requested analysis columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _save_figure(fig: Figure, name: str) -> Figure:
    """Save an analysis figure in the configured figures directory."""
    output_path = FIGURES_DIR / name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return fig


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize configured numeric features by flood class."""
    _require_columns(df, [TARGET, *NUMERIC_FEATURES])
    return df.groupby(TARGET)[list(NUMERIC_FEATURES)].agg(
        ["mean", "median", "std"]
    )


def landcover_by_class(df: pd.DataFrame) -> tuple[pd.DataFrame, Figure]:
    """Tabulate land cover by flood class and save its grouped bar chart."""
    _require_columns(df, ["land_cover_type", TARGET])
    table = pd.crosstab(df["land_cover_type"], df[TARGET])
    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax)
    ax.set(title="Land cover by flood class", xlabel="Land cover type", ylabel="Count")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return table, _save_figure(fig, "landcover_by_class.png")


def plot_distributions(df: pd.DataFrame, features: Sequence[str]) -> Figure:
    """Save overlaid flood-class histograms for the requested features."""
    _require_columns(df, [TARGET, *features])
    if not features:
        raise ValueError("features must contain at least one column.")

    fig, axes = plt.subplots(
        len(features), 1, figsize=(9, 4 * len(features)), squeeze=False
    )
    colors = {0: "blue", 1: "red"}
    for ax, feature in zip(axes[:, 0], features):
        for flood_class, color in colors.items():
            values = df.loc[df[TARGET].eq(flood_class), feature].dropna()
            ax.hist(values, bins=30, alpha=0.5, color=color, label=f"Flooded {flood_class}")
        ax.set(title=f"{feature} by flood class", xlabel=feature, ylabel="Count")
        ax.legend()
    fig.tight_layout()
    return _save_figure(fig, "feature_distributions.png")


def plot_correlation(df: pd.DataFrame, features: Sequence[str]) -> Figure:
    """Save a correlation heatmap for the requested features and flood target."""
    columns = list(dict.fromkeys([*features, TARGET]))
    _require_columns(df, columns)
    if not features:
        raise ValueError("features must contain at least one column.")

    correlation = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(7, len(columns)), max(6, len(columns) * 0.8)))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), labels=correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), labels=correlation.index)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Feature correlation including flood target")
    fig.tight_layout()
    return _save_figure(fig, "feature_correlation.png")


def plot_by_storm(df: pd.DataFrame) -> Figure:
    """Save rainfall boxplots grouped by storm and flood class."""
    required = ["storm_name", "rainfall_inches", TARGET]
    _require_columns(df, required)
    storms = sorted(df["storm_name"].dropna().unique())
    if not storms:
        raise ValueError("No storm names are available to plot.")

    fig, ax = plt.subplots(figsize=(max(9, len(storms) * 2.5), 6))
    positions: list[float] = []
    labels: list[str] = []
    colors = {0: "blue", 1: "red"}
    for index, storm in enumerate(storms):
        center = index * 3.0
        for offset, flood_class in ((-0.4, 0), (0.4, 1)):
            values = df.loc[
                df["storm_name"].eq(storm) & df[TARGET].eq(flood_class),
                "rainfall_inches",
            ].dropna()
            if not values.empty:
                box = ax.boxplot(values, positions=[center + offset], widths=0.65)
                for patch in box["boxes"]:
                    patch.set(color=colors[flood_class])
                positions.append(center + offset)
                labels.append(str(storm))
    centers = [index * 3.0 for index in range(len(storms))]
    ax.set_xticks(centers, labels=[str(storm) for storm in storms], rotation=30, ha="right")
    ax.set(title="Rainfall by storm and flood class", ylabel="Rainfall (inches)")
    ax.legend(
        handles=[
            Patch(facecolor=colors[0], label="Not flooded"),
            Patch(facecolor=colors[1], label="Flooded"),
        ]
    )
    fig.tight_layout()
    return _save_figure(fig, "rainfall_by_storm.png")


def plot_map(df: pd.DataFrame) -> Figure:
    """Save one location scatter map panel per storm with flood-class colors."""
    _require_columns(df, ["lon", "lat", "storm_name", TARGET])
    storms = sorted(df["storm_name"].dropna().unique())
    if not storms:
        raise ValueError("No storm names are available to plot.")

    fig, axes = plt.subplots(
        1, len(storms), figsize=(6 * len(storms), 6), squeeze=False, sharex=True, sharey=True
    )
    colors = {0: "blue", 1: "red"}
    for ax, storm in zip(axes[0], storms):
        storm_df = df.loc[df["storm_name"].eq(storm)]
        for flood_class, color in colors.items():
            points = storm_df.loc[storm_df[TARGET].eq(flood_class)]
            ax.scatter(points["lon"], points["lat"], c=color, s=5, alpha=0.55, label=f"Flooded {flood_class}")
        ax.set(title=str(storm), xlabel="Longitude", ylabel="Latitude")
        ax.legend()
    fig.suptitle("Flood locations by storm")
    fig.tight_layout()
    return _save_figure(fig, "flood_locations_by_storm.png")


def check_imperviousness(df: pd.DataFrame) -> None:
    """Print imperviousness ranges, class correlations, and a synthetic-data verdict."""
    _require_columns(df, ["land_cover_type", "imperviousness", "paved_pct"])
    unique_counts: list[int] = []
    correlations: list[float] = []

    for land_cover, group in df.groupby("land_cover_type", dropna=False):
        values = group["imperviousness"].dropna()
        unique_count = int(values.nunique())
        unique_counts.append(unique_count)
        minimum = values.min() if not values.empty else float("nan")
        maximum = values.max() if not values.empty else float("nan")
        print(
            f"{land_cover}: min={minimum}, max={maximum}, "
            f"unique imperviousness values={unique_count}"
        )

        if isinstance(land_cover, str) and land_cover.startswith("Developed"):
            correlation = group["imperviousness"].corr(group["paved_pct"])
            correlations.append(correlation)
            print(f"  imperviousness/paved_pct correlation={correlation:.3f}")

    developed_counts = [
        count
        for land_cover, count in zip(
            df.groupby("land_cover_type", dropna=False).size().index, unique_counts
        )
        if isinstance(land_cover, str) and land_cover.startswith("Developed")
    ]
    has_many_values = bool(developed_counts) and all(count >= 10 for count in developed_counts)
    has_nontrivial_correlation = any(
        pd.notna(correlation) and abs(correlation) >= 0.2
        for correlation in correlations
    )
    verdict = (
        "looks like real raster values"
        if has_many_values and has_nontrivial_correlation
        else "looks synthetic"
    )
    print(verdict)
