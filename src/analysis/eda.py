"""
src/analysis/eda.py
-------------------
Exploratory Data Analysis: distributions, correlations, category benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


NUMERIC_COLS = [
    "discounted_price", "actual_price", "discount_pct_100",
    "rating", "rating_count", "price_discount_amount",
    "discount_ratio", "log_rating_count",
]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return transposed describe() for all numeric columns."""
    return df[NUMERIC_COLS].describe().T


def plot_distributions(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Histogram + KDE for each numeric column."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for col in NUMERIC_COLS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/dist_{col}.png", dpi=100)
        plt.close(fig)
    print(f"[eda] Distribution plots saved to {output_dir}/")


def plot_boxplots(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Boxplots for price and discount columns."""
    cols = ["actual_price", "discounted_price", "price_discount_amount", "discount_pct_100"]
    for col in cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/box_{col}.png", dpi=100)
        plt.close(fig)


def correlation_heatmaps(df: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """Compute and plot Pearson & Spearman correlation matrices."""
    results = {}
    for method in ("pearson", "spearman"):
        corr = df[NUMERIC_COLS].corr(method=method)
        results[method] = corr
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        ax.set_title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/corr_{method}.png", dpi=100)
        plt.close(fig)
    return results


def category_insights(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Compute weighted category ranking:
      70% normalised avg rating + 30% normalised avg rating_count.
    """
    agg = df.groupby("category").agg(
        product_count=("product_id" if "product_id" in df.columns else "rating", "count"),
        avg_rating=("rating", "mean"),
        avg_rating_count=("rating_count", "mean"),
        avg_discount_pct=("discount_pct_100", "mean"),
    )
    agg["weighted_rank"] = (
        0.7 * agg["avg_rating"] / agg["avg_rating"].max()
        + 0.3 * agg["avg_rating_count"] / agg["avg_rating_count"].max()
    )
    return agg.sort_values("weighted_rank", ascending=False).head(top_n)


def price_quartile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Segment products into price quartiles and summarise."""
    df = df.copy()
    df["price_quartile"] = pd.qcut(
        df["actual_price"], q=4,
        labels=["Q1 Cheapest", "Q2", "Q3", "Q4 Expensive"],
    )
    return df.groupby("price_quartile")["actual_price"].describe()


def run_eda(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Run full EDA suite and print key summaries."""
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(descriptive_stats(df).to_string())

    plot_distributions(df, output_dir)
    plot_boxplots(df, output_dir)
    correlation_heatmaps(df, output_dir)

    print("\n=== TOP CATEGORIES (Weighted Rank) ===")
    print(category_insights(df).to_string())

    print("\n=== PRICE QUARTILE ANALYSIS ===")
    print(price_quartile_analysis(df).to_string())
