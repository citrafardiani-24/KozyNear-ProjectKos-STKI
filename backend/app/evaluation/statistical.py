"""Statistical significance tests untuk compare IR models.

Untuk per-query metric (e.g., AP per query untuk Model A vs Model B):
- **paired_ttest**: kalau distribusi roughly normal
- **wilcoxon_signed_rank**: non-parametric, lebih robust untuk metric bounded [0,1]

Convention: H0 = "Model A = Model B", alpha = 0.05.
Kalau p-value < 0.05, reject H0 => significant difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy import stats


@dataclass
class StatTestResult:
    """Hasil statistical test."""
    test_name: str
    statistic: float
    p_value: float
    n: int  # jumlah pair
    significant: bool  # p < alpha
    alpha: float = 0.05
    effect_size: float | None = None  # cohen's d untuk t-test

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, n={self.n}, {sig} (alpha={self.alpha})"
        )


def paired_ttest(
    model_a_scores: list[float],
    model_b_scores: list[float],
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> StatTestResult:
    """Paired t-test untuk per-query metric.

    Args:
        model_a_scores: metric per query untuk Model A (e.g., AP@10)
        model_b_scores: same query order, Model B
        alpha: significance threshold (default 0.05)
        alternative: 'two-sided' / 'less' / 'greater' (one-sided)

    Returns StatTestResult dengan Cohen's d effect size.
    """
    if len(model_a_scores) != len(model_b_scores):
        raise ValueError(
            f"Length mismatch: A={len(model_a_scores)}, B={len(model_b_scores)}"
        )
    if len(model_a_scores) < 2:
        raise ValueError("Butuh minimal 2 query untuk t-test")

    result = stats.ttest_rel(
        model_a_scores, model_b_scores, alternative=alternative
    )

    # Cohen's d effect size (paired)
    import numpy as np

    diffs = np.array(model_a_scores) - np.array(model_b_scores)
    effect_size = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else 0.0

    return StatTestResult(
        test_name="paired_ttest",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n=len(model_a_scores),
        significant=result.pvalue < alpha,
        alpha=alpha,
        effect_size=effect_size,
    )


def rank_biserial(
    model_a_scores: list[float], model_b_scores: list[float]
) -> float:
    """Effect size matched-pairs rank-biserial untuk Wilcoxon signed-rank.

    r = (W+ - W-) / (W+ + W-), dengan W+/- = jumlah rank |selisih| untuk
    pasangan A>B / A<B (selisih nol dibuang, konsisten Wilcoxon).
    Interpretasi kasar |r|: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium,
    >0.5 large. Positif = A cenderung lebih tinggi dari B.
    """
    from scipy.stats import rankdata

    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores) if a != b]
    if not diffs:
        return 0.0
    ranks = rankdata([abs(d) for d in diffs])
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total else 0.0


def bootstrap_ci_mean(
    values: list[float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI untuk mean metric per-query (mis. MAP).

    Resample query dengan penggantian; cocok untuk n kecil tanpa asumsi
    normalitas. Deterministik via seed.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    alpha_tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha_tail, 1.0 - alpha_tail])
    return (float(lo), float(hi))


@dataclass
class HolmEntry:
    """Hasil koreksi Holm-Bonferroni untuk satu uji dalam keluarga uji."""
    label: str
    p_value: float
    p_adjusted: float       # Holm step-down adjusted p-value (monotone)
    threshold: float        # ambang alpha/(m - rank) untuk uji ini
    significant: bool       # p_adjusted < alpha


def holm_bonferroni(
    tests: list[tuple[str, float]],
    alpha: float = 0.05,
) -> list[HolmEntry]:
    """Koreksi multiple comparison Holm-Bonferroni (step-down).

    Menjalankan m uji sekaligus pada alpha=0.05 menggelembungkan peluang
    false positive (m=6 -> ~26%). Holm: urutkan p ascending, bandingkan
    p_i dengan alpha/(m-i); berhenti di kegagalan pertama, sisanya tidak
    signifikan. Lebih powerful dari Bonferroni murni, tetap kontrol FWER.

    Args:
        tests: list (label, p_value), urutan bebas.
    Returns:
        list HolmEntry dengan urutan input dipertahankan.
    """
    m = len(tests)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: tests[i][1])

    # Adjusted p: max kumulatif dari (m - rank) * p, di-cap 1.0 (monotone)
    adjusted = [0.0] * m
    running_max = 0.0
    rejecting = True
    thresholds = [0.0] * m
    significant = [False] * m
    for rank, idx in enumerate(order):
        p = tests[idx][1]
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
        thresholds[idx] = alpha / (m - rank)
        # step-down: begitu satu gagal, semua setelahnya gagal
        if rejecting and p <= thresholds[idx]:
            significant[idx] = True
        else:
            rejecting = False

    return [
        HolmEntry(
            label=tests[i][0],
            p_value=tests[i][1],
            p_adjusted=adjusted[i],
            threshold=thresholds[i],
            significant=significant[i],
        )
        for i in range(m)
    ]


def wilcoxon_signed_rank(
    model_a_scores: list[float],
    model_b_scores: list[float],
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> StatTestResult:
    """Wilcoxon signed-rank test — non-parametric paired comparison.

    Lebih robust untuk metric bounded [0, 1] yang sering tidak normal.
    Direkomendasikan untuk IR metric comparison.
    """
    if len(model_a_scores) != len(model_b_scores):
        raise ValueError("Length mismatch")
    if len(model_a_scores) < 5:
        raise ValueError("Butuh minimal 5 query untuk Wilcoxon (statistical power)")

    result = stats.wilcoxon(
        model_a_scores, model_b_scores, alternative=alternative
    )

    return StatTestResult(
        test_name="wilcoxon_signed_rank",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n=len(model_a_scores),
        significant=result.pvalue < alpha,
        alpha=alpha,
        effect_size=None,  # Wilcoxon gak punya standard effect size
    )
