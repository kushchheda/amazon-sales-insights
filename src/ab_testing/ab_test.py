"""
src/ab_testing/ab_test.py
--------------------------
A/B testing engine: t-tests, bootstrap CI, power analysis, bandit simulation,
chi-square tests, and business interpretation.

Hypothesis
----------
Products with higher discounts (>50%) generate more customer engagement
(rating_count) than products with lower discounts (<50%).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

from src.utils.stats_utils import (
    two_sample_ttest,
    bootstrap_ci,
    power_analysis,
    required_sample_size,
)


# ---------------------------------------------------------------------------
# Group Assignment
# ---------------------------------------------------------------------------

def assign_discount_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add discount_group column: 'Low Discount' (<= 50%) / 'High Discount' (> 50%)."""
    df = df.copy()
    df["discount_group"] = pd.cut(
        df["discount_pct_100"],
        bins=[-0.01, 50, 100],
        labels=["Low Discount", "High Discount"],
    )
    return df


# ---------------------------------------------------------------------------
# Core A/B Test
# ---------------------------------------------------------------------------

def run_engagement_ab_test(df: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """
    Primary A/B test: Low Discount (A) vs High Discount (B) on log(1 + rating_count).
    Returns a results dict with statistics and uplift figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = assign_discount_groups(df)

    group_a = df[df["discount_group"] == "Low Discount"]["log_rating_count"].dropna().values
    group_b = df[df["discount_group"] == "High Discount"]["log_rating_count"].dropna().values

    ttest = two_sample_ttest(group_a, group_b)
    ci = bootstrap_ci(group_a, group_b)

    raw_a = df[df["discount_group"] == "Low Discount"]["rating_count"].dropna().values
    raw_b = df[df["discount_group"] == "High Discount"]["rating_count"].dropna().values
    uplift_raw_pct = (raw_b.mean() - raw_a.mean()) / raw_a.mean() * 100

    power = power_analysis(ttest["cohens_d"], n=min(len(group_a), len(group_b)))
    n_required = required_sample_size(
        baseline_mean=group_a.mean(), mde=0.15, std=group_a.std()
    )

    results = {
        "n_a": len(group_a), "n_b": len(group_b),
        "mean_log_a": group_a.mean(), "mean_log_b": group_b.mean(),
        "mean_raw_a": raw_a.mean(), "mean_raw_b": raw_b.mean(),
        "uplift_raw_pct": uplift_raw_pct,
        **ttest,
        **{f"ci_{k}": v for k, v in ci.items()},
        "achieved_power": power,
        "required_n_per_group": n_required,
    }

    _plot_ab_boxplot(df, output_dir)
    _print_ab_summary(results)
    return results


def _plot_ab_boxplot(df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="discount_group", y="log_rating_count", ax=ax)
    ax.set_title("A/B Test: Engagement (log rating count) by Discount Group")
    ax.set_ylabel("log(1 + rating_count)")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/ab_engagement_boxplot.png", dpi=100)
    plt.close(fig)


def _print_ab_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS — ENGAGEMENT (log rating_count)")
    print("=" * 60)
    print(f"  Group A (Low Discount):  n={r['n_a']:,}, mean log count={r['mean_log_a']:.4f}")
    print(f"  Group B (High Discount): n={r['n_b']:,}, mean log count={r['mean_log_b']:.4f}")
    print(f"  Raw uplift:              {r['uplift_raw_pct']:+.1f}%")
    print(f"  p-value:                 {r['p_value']:.4f} {'✅ Significant' if r['significant'] else '❌ Not Significant'}")
    print(f"  Cohen's d:               {r['cohens_d']:.4f}")
    print(f"  Achieved power:          {r['achieved_power']:.4f}")
    print(f"  Bootstrap 95% CI:        [{r['ci_rel_lower_pct']:.1f}%, {r['ci_rel_upper_pct']:.1f}%]")
    print(f"  Required n/group (15% MDE, 80% power): {r['required_n_per_group']:,}")
    print(f"  ⚠️  Current n is {'adequate' if min(r['n_a'], r['n_b']) >= r['required_n_per_group'] else 'underpowered'}")


# ---------------------------------------------------------------------------
# Rating T-tests (secondary checks)
# ---------------------------------------------------------------------------

def run_rating_ttests(df: pd.DataFrame) -> None:
    """T-tests: discount impact and price impact on rating."""
    df = assign_discount_groups(df)

    low_r = df[df["discount_group"] == "Low Discount"]["rating"].dropna().values
    high_r = df[df["discount_group"] == "High Discount"]["rating"].dropna().values
    res_disc = two_sample_ttest(low_r, high_r)

    median_price = df["actual_price"].median()
    low_p = df[df["actual_price"] < median_price]["rating"].dropna().values
    high_p = df[df["actual_price"] >= median_price]["rating"].dropna().values
    res_price = two_sample_ttest(low_p, high_p)

    print("\n=== T-TEST: Discount → Rating ===")
    print(f"  Low Discount mean: {low_r.mean():.4f} | High Discount mean: {high_r.mean():.4f}")
    print(f"  p={res_disc['p_value']:.4f} | d={res_disc['cohens_d']:.4f} | {'Significant' if res_disc['significant'] else 'Not Significant'}")

    print("\n=== T-TEST: Price → Rating ===")
    print(f"  Low Price mean: {low_p.mean():.4f} | High Price mean: {high_p.mean():.4f}")
    print(f"  p={res_price['p_value']:.4f} | d={res_price['cohens_d']:.4f} | {'Significant' if res_price['significant'] else 'Not Significant'}")


# ---------------------------------------------------------------------------
# Chi-square Tests
# ---------------------------------------------------------------------------

def run_chi_square_tests(df: pd.DataFrame) -> None:
    """Chi-square independence tests across categorical pairs."""
    df = df.copy()
    df["rating_level"] = pd.qcut(df["rating"], q=2, labels=["Low", "High"])
    df["discount_level"] = pd.qcut(df["discount_pct_100"], q=3, labels=["Low", "Medium", "High"])
    df["rating_count_level"] = pd.qcut(df["rating_count"], q=3, labels=["Low", "Medium", "High"])
    df["price_level"] = pd.qcut(df["actual_price"], q=3, labels=["Low", "Medium", "High"])

    pairs = [
        ("discount_level", "rating_level"),
        ("price_level", "rating_level"),
        ("rating_count_level", "rating_level"),
        ("category", "rating_level"),
        ("category", "discount_level"),
    ]

    print("\n=== CHI-SQUARE TESTS ===")
    for col1, col2 in pairs:
        table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, _ = chi2_contingency(table)
        sig = "✅ Significant" if p < 0.05 else "❌ Not Significant"
        print(f"  {col1} × {col2}: χ²={chi2:.2f}, p={p:.4f}, dof={dof} → {sig}")


# ---------------------------------------------------------------------------
# Multi-Armed Bandit
# ---------------------------------------------------------------------------

def simulate_epsilon_greedy_bandit(
    df: pd.DataFrame,
    n_rounds: int = 1000,
    epsilon: float = 0.1,
) -> dict:
    """
    ε-greedy bandit simulation over Low/High discount groups.
    Returns allocation counts, average rewards, and total reward.
    """
    df = assign_discount_groups(df)
    group_data = {
        "Low Discount": df[df["discount_group"] == "Low Discount"]["rating_count"].dropna().values,
        "High Discount": df[df["discount_group"] == "High Discount"]["rating_count"].dropna().values,
    }
    groups = list(group_data.keys())
    rng = np.random.default_rng(42)

    rewards = {g: [] for g in groups}
    selections = {g: 0 for g in groups}

    for _ in range(n_rounds):
        if rng.random() < epsilon or all(len(rewards[g]) == 0 for g in groups):
            chosen = rng.choice(groups)
        else:
            chosen = max(groups, key=lambda g: np.mean(rewards[g]) if rewards[g] else 0)

        reward = rng.choice(group_data[chosen])
        rewards[chosen].append(reward)
        selections[chosen] += 1

    total_reward = sum(sum(v) for v in rewards.values())
    avg_reward = total_reward / n_rounds

    print("\n=== MULTI-ARMED BANDIT (ε-greedy) ===")
    for g in groups:
        print(f"  {g}: selected {selections[g]:,}×, avg reward {np.mean(rewards[g]):.0f}")
    print(f"  Total reward: {total_reward:,.0f} | Avg per round: {avg_reward:,.0f}")

    return {"selections": selections, "avg_reward": avg_reward, "total_reward": total_reward}
