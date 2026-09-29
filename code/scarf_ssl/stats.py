"""
Statistical significance testing (Section 2.6.1): paired t-test and
Wilcoxon signed-rank test on per-fold accuracy vectors.
"""

from __future__ import annotations

from typing import Dict, List

from scipy import stats


def paired_significance(acc_a: List[float], acc_b: List[float]) -> Dict[str, float]:
    """Paired t-test and Wilcoxon signed-rank test comparing two methods'
    per-fold accuracy vectors (same folds, same order).

    NOTE: with n=5 paired folds, the Wilcoxon signed-rank test's minimum
    achievable two-sided p-value is 0.0625, regardless of effect size — it
    can never reach the conventional p<0.05 threshold with only 5 folds.
    It's reported here for transparency/completeness alongside the t-test,
    but the t-test is the practically informative test at this sample
    size; don't over-interpret a non-significant Wilcoxon result on its
    own with n=5.
    """
    t_stat, t_p = stats.ttest_rel(acc_a, acc_b)
    try:
        w_stat, w_p = stats.wilcoxon(acc_a, acc_b)
    except ValueError:
        # Raised when all paired differences are zero.
        w_stat, w_p = float("nan"), float("nan")
    return {
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }
