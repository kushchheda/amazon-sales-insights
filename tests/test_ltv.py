"""
tests/test_ltv.py
-----------------
Unit tests for LTV modeling logic.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ltv.ltv_model import build_ltv_tables, clv_segments, run_scenario_analysis
from src.utils.stats_utils import cohens_d, bootstrap_ci, required_sample_size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal clean DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 100
    actual_price = rng.uniform(100, 5000, n)
    discount_pct = rng.uniform(10, 70, n)
    discounted_price = actual_price * (1 - discount_pct / 100)
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "product_name": [f"Product {i}" for i in range(n)],
        "category": rng.choice(["electronics", "kitchen", "books"], n),
        "actual_price": actual_price,
        "discounted_price": discounted_price,
        "discount_pct_100": discount_pct,
        "discount_percentage": discount_pct / 100,
        "rating": rng.uniform(3.0, 5.0, n),
        "rating_count": rng.integers(100, 50000, n).astype(float),
        "log_rating_count": np.log1p(rng.integers(100, 50000, n)),
        "price_discount_amount": actual_price - discounted_price,
        "discount_ratio": discounted_price / actual_price,
    })


# ---------------------------------------------------------------------------
# LTV Tests
# ---------------------------------------------------------------------------

class TestBuildLtvTables:
    def test_output_shapes(self, sample_df):
        df_ltv, cat_ltv = build_ltv_tables(sample_df)
        assert len(df_ltv) == len(sample_df)
        assert len(cat_ltv) == sample_df["category"].nunique()

    def test_clv_positive(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df)
        assert (df_ltv["clv_per_customer"] > 0).all()

    def test_product_total_clv_positive(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df)
        assert (df_ltv["product_total_clv"] > 0).all()

    def test_higher_margin_raises_clv(self, sample_df):
        df_base, _ = build_ltv_tables(sample_df, gross_margin_pct=0.30)
        df_high, _ = build_ltv_tables(sample_df, gross_margin_pct=0.40)
        assert df_high["clv_per_customer"].mean() > df_base["clv_per_customer"].mean()

    def test_higher_retention_raises_clv(self, sample_df):
        df_base, _ = build_ltv_tables(sample_df, base_retention_rate=0.60)
        df_high, _ = build_ltv_tables(sample_df, base_retention_rate=0.70)
        assert df_high["clv_per_customer"].mean() > df_base["clv_per_customer"].mean()

    def test_retention_capped_at_095(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df, base_retention_rate=0.99)
        assert (df_ltv["retention_rate"] <= 0.95).all()


class TestClvSegments:
    def test_four_segments(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df)
        seg = clv_segments(df_ltv)
        assert len(seg) == 4

    def test_ltv_pct_sums_to_one(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df)
        seg = clv_segments(df_ltv)
        assert abs(seg["ltv_pct"].sum() - 1.0) < 1e-6

    def test_very_high_segment_high_ltv(self, sample_df):
        df_ltv, _ = build_ltv_tables(sample_df)
        seg = clv_segments(df_ltv)
        assert seg.loc["Very High", "ltv_pct"] > seg.loc["Low", "ltv_pct"]


class TestScenarioAnalysis:
    def test_returns_three_scenarios(self, sample_df):
        result = run_scenario_analysis(sample_df)
        assert len(result) == 3

    def test_optimistic_beats_conservative(self, sample_df):
        result = run_scenario_analysis(sample_df)
        assert result.loc["Optimistic", "Avg CLV / Customer"] > result.loc["Conservative", "Avg CLV / Customer"]


# ---------------------------------------------------------------------------
# Stats Utils Tests
# ---------------------------------------------------------------------------

class TestStatsUtils:
    def test_cohens_d_zero_for_same_groups(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_cohens_d_positive_when_b_larger(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert cohens_d(a, b) > 0

    def test_bootstrap_ci_contains_zero_for_same_groups(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 200)
        result = bootstrap_ci(a, a, n_bootstrap=500)
        assert result["abs_lower"] <= 0 <= result["abs_upper"]

    def test_required_sample_size_positive(self):
        n = required_sample_size(baseline_mean=10.0, mde=0.10, std=2.0)
        assert n > 0

    def test_larger_mde_requires_fewer_samples(self):
        n_small_mde = required_sample_size(baseline_mean=10.0, mde=0.05, std=2.0)
        n_large_mde = required_sample_size(baseline_mean=10.0, mde=0.20, std=2.0)
        assert n_small_mde > n_large_mde
