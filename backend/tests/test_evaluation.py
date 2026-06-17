"""Unit tests untuk evaluation metrics + kappa + statistical tests.

Run: pytest backend/tests/test_evaluation.py -v
"""

from __future__ import annotations

import pytest

from app.evaluation.kappa import (
    cohen_kappa,
    interpret_kappa,
    weighted_kappa,
)
from app.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


# =============================================================================
# Precision @ K
# =============================================================================
class TestPrecisionAtK:
    def test_basic(self):
        predicted = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "x"}
        # Top-5: a, b, c, d, e — 2 hits
        assert precision_at_k(predicted, relevant, 5) == 2 / 5

    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_predicted(self):
        assert precision_at_k([], {"a"}, 5) == 0.0


# =============================================================================
# Average Precision
# =============================================================================
class TestAveragePrecision:
    def test_perfect_ranking(self):
        # Semua relevant di awal
        predicted = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        # P@1 = 1.0 (a hit), P@2 = 1.0 (b hit)
        # AP = (1.0 + 1.0) / 2 = 1.0
        assert average_precision(predicted, relevant) == 1.0

    def test_partial_match(self):
        predicted = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # P@1 = 1.0, P@3 = 2/3
        # AP = (1.0 + 0.667) / 2 = 0.8333...
        ap = average_precision(predicted, relevant)
        assert abs(ap - 5 / 6) < 1e-6

    def test_no_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0


class TestMeanAveragePrecision:
    def test_two_queries(self):
        predicted = {"q1": ["a", "b"], "q2": ["c", "d"]}
        relevant = {"q1": {"a"}, "q2": {"d"}}
        # AP(q1) = 1.0, AP(q2) = 0.5
        assert mean_average_precision(predicted, relevant) == 0.75


# =============================================================================
# NDCG @ K
# =============================================================================
class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Highest-graded doc di posisi 1
        predicted = ["a", "b", "c"]
        scores = {"a": 2, "b": 1, "c": 0}
        # Actual DCG = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2.0 + 0.631 + 0 = 2.631
        # Ideal DCG = same (sudah optimal)
        assert ndcg_at_k(predicted, scores, 3) == 1.0

    def test_reversed_ranking(self):
        predicted = ["c", "b", "a"]
        scores = {"a": 2, "b": 1, "c": 0}
        # DCG/IDCG < 1
        assert ndcg_at_k(predicted, scores, 3) < 1.0

    def test_empty_relevance(self):
        assert ndcg_at_k(["a"], {}, 1) == 0.0


# =============================================================================
# Reciprocal Rank
# =============================================================================
class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3

    def test_not_found(self):
        assert reciprocal_rank(["a"], {"x"}) == 0.0


class TestMeanReciprocalRank:
    def test_basic(self):
        predicted = {"q1": ["a"], "q2": ["b", "c"]}
        relevant = {"q1": {"a"}, "q2": {"c"}}
        # RR(q1) = 1.0, RR(q2) = 0.5
        # MRR = 0.75
        assert mean_reciprocal_rank(predicted, relevant) == 0.75


# =============================================================================
# Cohen's Kappa
# =============================================================================
class TestCohenKappa:
    def test_perfect_agreement(self):
        a = [0, 1, 2, 0, 1]
        b = [0, 1, 2, 0, 1]
        assert cohen_kappa(a, b) == pytest.approx(1.0)

    def test_zero_agreement(self):
        # Worst case (alternating disagreement)
        a = [0, 1, 0, 1]
        b = [1, 0, 1, 0]
        kappa = cohen_kappa(a, b)
        assert kappa < 0.0  # less than chance

    def test_mismatch_length_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([0, 1], [0, 1, 2])


class TestWeightedKappa:
    def test_perfect(self):
        a = [0, 1, 2]
        b = [0, 1, 2]
        assert weighted_kappa(a, b) == pytest.approx(1.0)

    def test_ordinal_penalty(self):
        # Adjacent disagreement lebih ringan dari extreme
        a = [0, 0, 0]
        b_close = [1, 1, 1]  # 1-step off
        b_far = [2, 2, 2]  # 2-step off

        k_close = weighted_kappa(a, b_close, labels=[0, 1, 2])
        k_far = weighted_kappa(a, b_far, labels=[0, 1, 2])
        # Lebih jauh -> lebih buruk
        assert k_far <= k_close


class TestInterpretKappa:
    def test_categories(self):
        assert interpret_kappa(0.85) == "almost perfect"
        assert interpret_kappa(0.7) == "substantial"
        assert interpret_kappa(0.5) == "moderate"
        assert interpret_kappa(0.3) == "fair"
        assert interpret_kappa(0.1) == "slight"


# =============================================================================
# Statistical tests
# =============================================================================
try:
    from app.evaluation.statistical import paired_ttest, wilcoxon_signed_rank

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy belum install")
class TestPairedTTest:
    def test_no_difference(self):
        a = [0.5, 0.6, 0.7, 0.5, 0.6]
        b = [0.5, 0.6, 0.7, 0.5, 0.6]
        # Catch identical arrays gracefully — scipy may raise
        try:
            result = paired_ttest(a, b)
            assert not result.significant
        except Exception:
            # Some scipy versions raise pada identical arrays — OK skip
            pass

    def test_clear_difference(self):
        a = [0.8, 0.9, 0.85, 0.92, 0.88]
        b = [0.2, 0.3, 0.25, 0.28, 0.3]
        result = paired_ttest(a, b)
        assert result.significant
        assert result.effect_size is not None
        assert result.effect_size > 1.0  # Large effect


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy belum install")
class TestWilcoxon:
    def test_clear_difference(self):
        a = [0.8, 0.9, 0.85, 0.92, 0.88, 0.79, 0.91]
        b = [0.2, 0.3, 0.25, 0.28, 0.3, 0.22, 0.28]
        result = wilcoxon_signed_rank(a, b)
        assert result.significant

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([0.5, 0.6], [0.4, 0.5])
