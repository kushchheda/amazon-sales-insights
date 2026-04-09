"""
src/ltv/ltv_model.py
--------------------
Customer Lifetime Value (CLV) modeling using an infinite-horizon formula.

Assumptions
-----------
- Each rating_count entry ≈ one unique customer
- Revenue per purchase ≈ discounted_price
- Gross margin is a fixed percentage
- Purchase frequency and retention scale with product rating
- Infinite-horizon CLV: CLV = (margin * freq) / (1 + discount_rate - retention)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Core LTV Builder
# ---------------------------------------------------------------------------

def build_ltv_tables(
    df_clean: pd.DataFrame,
    gross_margin_pct: float = 0.30,
    base_purchase_freq_per_year: float = 1.0,
    base_retention_rate: float = 0.60,
    discount_rate: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute product-level and category-level LTV tables.

    Returns
    -------
    df_ltv : product-level DataFrame with CLV columns
    category_ltv : category-level aggregated LTV DataFrame
    """
    df_ltv = df_clean.copy()

    # Unit margin
    df_ltv["unit_margin"] = df_ltv["discounted_price"] * gross_margin_pct

    # Rating-adjusted purchase frequency (higher-rated → more repeat purchases)
    rating_norm = df_ltv["rating"] / 5.0
    df_ltv["purchase_freq"] = base_purchase_freq_per_year * (0.5 + rating_norm)

    # Rating-adjusted retention rate (capped at 0.95)
    df_ltv["retention_rate"] = np.minimum(
        base_retention_rate * (0.7 + 0.6 * rating_norm), 0.95
    )

    # Infinite-horizon CLV per customer
    denom = 1 + discount_rate - df_ltv["retention_rate"]
    df_ltv["clv_per_customer"] = (
        df_ltv["unit_margin"] * df_ltv["purchase_freq"] / denom.clip(lower=0.01)
    )

    # Estimated customers ≈ rating_count
    df_ltv["estimated_customers"] = df_ltv["rating_count"]

    # Total product LTV
    df_ltv["product_total_clv"] = df_ltv["clv_per_customer"] * df_ltv["estimated_customers"]

    # Category-level aggregation
    category_ltv = (
        df_ltv.groupby("category")
        .agg(
            total_LTV=("product_total_clv", "sum"),
            avg_CLV_per_customer=("clv_per_customer", "mean"),
            total_customers=("estimated_customers", "sum"),
        )
        .sort_values("total_LTV", ascending=False)
    )

    return df_ltv, category_ltv


# ---------------------------------------------------------------------------
# Lorenz Curve & Pareto Analysis
# ---------------------------------------------------------------------------

def lorenz_pareto_analysis(df_ltv: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """
    Compute Lorenz curve and identify what share of customers drives 80% of LTV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_sorted = df_ltv.sort_values("clv_per_customer", ascending=False).copy()

    total_customers = df_sorted["estimated_customers"].sum()
    total_ltv = df_sorted["product_total_clv"].sum()

    df_sorted["cum_customer_pct"] = df_sorted["estimated_customers"].cumsum() / total_customers
    df_sorted["cum_ltv_pct"] = df_sorted["product_total_clv"].cumsum() / total_ltv

    # Find LTV share at top-25% customers
    top25_mask = df_sorted["cum_customer_pct"] <= 0.25
    ltv_share_top25 = df_sorted.loc[top25_mask, "product_total_clv"].sum() / total_ltv

    # Plot Lorenz curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_sorted["cum_customer_pct"], df_sorted["cum_ltv_pct"], label="Lorenz Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Equality")
    ax.axvline(0.25, color="red", linestyle=":", label="Top 25% customers")
    ax.set_xlabel("Cumulative Customer Share")
    ax.set_ylabel("Cumulative LTV Share")
    ax.set_title("Lorenz Curve — LTV Concentration")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{output_dir}/lorenz_curve.png", dpi=100)
    plt.close(fig)

    print(f"\n[ltv] Top 25% customers → {ltv_share_top25:.1%} of total LTV (Pareto effect)")
    return {"ltv_share_top25_pct": ltv_share_top25 * 100}


# ---------------------------------------------------------------------------
# CLV Segmentation
# ---------------------------------------------------------------------------

def clv_segments(df_ltv: pd.DataFrame) -> pd.DataFrame:
    """Assign Low / Medium / High / Very High CLV segments and summarise."""
    df_seg = df_ltv.copy()
    df_seg["clv_segment"] = pd.qcut(
        df_seg["clv_per_customer"],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
    )
    summary = df_seg.groupby("clv_segment").agg(
        total_customers=("estimated_customers", "sum"),
        total_LTV=("product_total_clv", "sum"),
    )
    summary["customer_pct"] = summary["total_customers"] / summary["total_customers"].sum()
    summary["ltv_pct"] = summary["total_LTV"] / summary["total_LTV"].sum()
    return summary


# ---------------------------------------------------------------------------
# Scenario Analysis
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Conservative": {"margin": 0.20, "retention": 0.50},
    "Base Case":    {"margin": 0.30, "retention": 0.60},
    "Optimistic":   {"margin": 0.40, "retention": 0.70},
}


def run_scenario_analysis(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compare mean CLV across Conservative / Base Case / Optimistic scenarios."""
    rows = []
    for name, params in SCENARIOS.items():
        df_sc, _ = build_ltv_tables(
            df_clean,
            gross_margin_pct=params["margin"],
            base_retention_rate=params["retention"],
        )
        rows.append({
            "Scenario": name,
            "Margin": f"{params['margin']:.0%}",
            "Retention": f"{params['retention']:.0%}",
            "Avg CLV / Customer": df_sc["clv_per_customer"].mean(),
            "Median CLV / Customer": df_sc["clv_per_customer"].median(),
            "Total LTV (M)": df_sc["product_total_clv"].sum() / 1e6,
        })
    df_scenarios = pd.DataFrame(rows).set_index("Scenario")
    print("\n=== SCENARIO ANALYSIS ===")
    print(df_scenarios.to_string())
    return df_scenarios


# ---------------------------------------------------------------------------
# Business Impact Print
# ---------------------------------------------------------------------------

def print_business_impact(df_ltv: pd.DataFrame, new_customers: int = 10_000) -> None:
    avg_clv = df_ltv["clv_per_customer"].mean()
    total_value = new_customers * avg_clv

    threshold = df_ltv["clv_per_customer"].quantile(0.75)
    high_clv_avg = df_ltv[df_ltv["clv_per_customer"] >= threshold]["clv_per_customer"].mean()
    targeted_value = new_customers * high_clv_avg
    uplift_pct = (targeted_value - total_value) / total_value * 100

    print("\n" + "=" * 60)
    print("💰 BUSINESS IMPACT TRANSLATION")
    print("=" * 60)
    print(f"  Acquiring {new_customers:,} average customers → ${total_value:,.0f} total LTV")
    print(f"  Targeting top-25% high-CLV customers → ${targeted_value:,.0f} total LTV")
    print(f"  Uplift from targeting high-CLV segment: +{uplift_pct:.0f}%")
    print(f"  CLV threshold to enter top 25%: ${threshold:,.2f} / customer")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ltv_distributions(df_ltv: pd.DataFrame, output_dir: str = "outputs") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for col, title in [
        ("clv_per_customer", "CLV per Customer"),
        ("product_total_clv", "Total Product LTV"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_ltv[col].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {title}")
        ax.set_xlabel(title)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/ltv_dist_{col}.png", dpi=100)
        plt.close(fig)


def run_ltv_pipeline(df_clean: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """Run the full LTV pipeline and return all key outputs."""
    df_ltv, category_ltv = build_ltv_tables(df_clean)

    print("\n=== TOP PRODUCTS BY CLV ===")
    cols = ["product_id", "product_name", "category", "discounted_price",
            "rating", "rating_count", "clv_per_customer", "product_total_clv"]
    cols = [c for c in cols if c in df_ltv.columns]
    print(df_ltv.sort_values("product_total_clv", ascending=False)[cols].head(15).to_string())

    print("\n=== TOP CATEGORIES BY LTV ===")
    print(category_ltv.head(15).to_string())

    plot_ltv_distributions(df_ltv, output_dir)
    pareto = lorenz_pareto_analysis(df_ltv, output_dir)
    segments = clv_segments(df_ltv)
    print("\n=== CLV SEGMENTS ===")
    print(segments.to_string())

    scenarios = run_scenario_analysis(df_clean)
    print_business_impact(df_ltv)

    return {
        "df_ltv": df_ltv,
        "category_ltv": category_ltv,
        "segments": segments,
        "scenarios": scenarios,
        "pareto": pareto,
    }
