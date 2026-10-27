"""Phase 1: load the California Housing dataset, inspect it, and save exploratory charts.

Run from the project root:  python scripts/explore.py
Charts are written to reports/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from homevalue.data import FEATURES, FIGURES_DIR, TARGET, load_dataset


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    df = load_dataset()

    section("Shape")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns "
          f"({len(FEATURES)} features + 1 target '{TARGET}')")
    print("Each row is one census block group, not a single house.")

    section("First rows")
    print(df.head().to_string())

    section("Column types")
    print(df.dtypes.to_string())

    section("Missing values per column")
    print(df.isna().sum().to_string())

    section("Summary statistics")
    print(df.describe().T.round(3).to_string())

    section("Target variable: MedHouseVal (units of $100,000)")
    capped = (df[TARGET] >= 5.0).sum()
    print(f"Mean ${df[TARGET].mean() * 100_000:,.0f}, "
          f"median ${df[TARGET].median() * 100_000:,.0f}")
    print(f"{capped:,} rows ({capped / len(df):.1%}) sit at the $500,001 cap; "
          "the model cannot learn values above it.")

    section("Correlation of each feature with the target")
    corr = df.corr()
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Distribution of every column. AveRooms, AveBedrms, Population and AveOccup have long
    # right tails, so they are clipped at the 99th percentile to keep the bulk readable.
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.flat, df.columns):
        values = df[col].clip(upper=df[col].quantile(0.99))
        sns.histplot(values, bins=50, ax=ax, color="#3b6ea5")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Feature and target distributions (clipped at 99th percentile)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sample = df.sample(5000, random_state=0)
    sns.scatterplot(data=sample, x="MedInc", y=TARGET, alpha=0.3, s=12, ax=ax)
    ax.set_title("Median income vs. median house value (strongest predictor)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "income_vs_value.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    points = ax.scatter(df["Longitude"], df["Latitude"], c=df[TARGET], cmap="viridis",
                        s=3, alpha=0.5)
    fig.colorbar(points, ax=ax, label="Median house value ($100k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("House value by location: coastal areas are pricier")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_value_map.png", dpi=120)
    plt.close(fig)

    section("Charts saved")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path.relative_to(FIGURES_DIR.parent.parent))


if __name__ == "__main__":
    main()
