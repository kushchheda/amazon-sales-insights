"""
src/utils/stats_utils.py
------------------------
Shared statistical helpers used across modules.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import ttest_power, tt_ind_solve_power


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Pooled Cohen's d effect size (b - a)."""
    na, nb = len(group_a), len(group_b)
    va, vb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    s_pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(group_b) - np.mean(group_a)) / s_pooled


def two_sample_ttest(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Run Welch's t-test and return a result dictionary."""
    t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
    d = cohens_d(group_a, group_b)
    return {"t_stat": t_stat, "p_value": p_val, "cohens_d": d, "significant": p_val < 0.05}


def bootstrap_ci(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95,
) -> dict:
    """
    Bootstrap confidence interval for difference in means (B - A).
    Returns absolute and relative (%) CI bounds.
    """
    rng = np.random.default_rng(42)
    diffs = [
        rng.choice(group_b, size=len(group_b), replace=True).mean()
        - rng.choice(group_a, size=len(group_a), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    lo = np.percentile(diffs, (100 - ci) / 2)
    hi = np.percentile(diffs, 100 - (100 - ci) / 2)
    base = np.mean(group_a)
    return {
        "abs_lower": lo,
        "abs_upper": hi,
        "rel_lower_pct": lo / base * 100,
        "rel_upper_pct": hi / base * 100,
        "point_estimate": np.mean(diffs),
    }


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Return achieved statistical power for a given effect size and sample size."""
    return ttest_power(effect_size=abs(effect_size), nobs=n, alpha=alpha, alternative="two-sided")


def required_sample_size(
    baseline_mean: float,
    mde: float,
    std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required sample size per group.
    mde: minimum detectable effect as a fraction of baseline_mean (e.g. 0.10 = 10%).
    """
    effect_size = (baseline_mean * mde) / std
    n = tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))
