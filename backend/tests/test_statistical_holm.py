"""Test koreksi Holm-Bonferroni."""
from app.evaluation.statistical import holm_bonferroni


def test_holm_known_case_eval_pvalues():
    """Kasus nyata eval kita: 6 uji, raw 4 signifikan, Holm 3 signifikan."""
    tests = [
        ("tfidf-bm25", 0.1205),
        ("tfidf-indobert", 0.0020),
        ("tfidf-hybrid", 0.3028),
        ("bm25-indobert", 0.0067),
        ("bm25-hybrid", 0.0302),
        ("indobert-hybrid", 0.0010),
    ]
    out = {e.label: e for e in holm_bonferroni(tests, alpha=0.05)}
    assert out["indobert-hybrid"].significant       # 0.0010 < 0.05/6
    assert out["tfidf-indobert"].significant        # 0.0020 < 0.05/5
    assert out["bm25-indobert"].significant         # 0.0067 < 0.05/4
    assert not out["bm25-hybrid"].significant       # 0.0302 > 0.05/3 = 0.0167
    assert not out["tfidf-bm25"].significant
    assert not out["tfidf-hybrid"].significant
    assert sum(e.significant for e in out.values()) == 3


def test_holm_adjusted_p_monotone_and_capped():
    tests = [("a", 0.04), ("b", 0.04), ("c", 0.04)]
    entries = holm_bonferroni(tests)
    adj = [e.p_adjusted for e in entries]
    assert all(0.0 <= p <= 1.0 for p in adj)
    # urutan sama -> adjusted sama (monotone running max)
    assert adj[0] == adj[1] == adj[2] == 0.12
    assert not any(e.significant for e in entries)  # 0.04 > 0.05/3


def test_holm_step_down_stops_at_first_failure():
    # p kecil banget lalu satu gagal -> sisanya gagal walau p < alpha
    tests = [("a", 0.001), ("b", 0.03), ("c", 0.04)]
    out = {e.label: e for e in holm_bonferroni(tests)}
    assert out["a"].significant            # 0.001 < 0.05/3
    assert not out["b"].significant        # 0.03 > 0.05/2 = 0.025 -> stop
    assert not out["c"].significant        # ikut gagal (step-down)


def test_holm_empty_and_single():
    assert holm_bonferroni([]) == []
    single = holm_bonferroni([("x", 0.03)])
    assert single[0].significant  # m=1 -> threshold alpha penuh
