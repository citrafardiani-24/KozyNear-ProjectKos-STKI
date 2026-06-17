"""Unit test Constraint Satisfaction @K (lensa eval smart pipeline)."""
from app.evaluation.metrics import constraint_satisfaction_at_k


def test_all_constraints_pass():
    res = [{"tipe": "putri", "harga_per_bulan": 700000,
            "fasilitas": ["ac", "wifi"], "lat": -5.37, "lng": 105.244}]
    c = {"gender": "putri", "harga_max": 1_000_000,
         "fasilitas": ["ac"], "anchor": (-5.3692, 105.2433)}
    assert constraint_satisfaction_at_k(res, c, k=5) == 1.0


def test_gender_violation_fails():
    res = [{"tipe": "putra", "harga_per_bulan": 700000,
            "fasilitas": ["ac"], "lat": -5.37, "lng": 105.244}]
    c = {"gender": "putri", "fasilitas": [], "anchor": None}
    assert constraint_satisfaction_at_k(res, c, k=5) == 0.0


def test_distance_violation_fails():
    # listing jauh dari anchor (> 3km)
    res = [{"tipe": "putri", "harga_per_bulan": 700000,
            "fasilitas": ["ac"], "lat": -5.41, "lng": 105.32}]
    c = {"gender": "putri", "fasilitas": ["ac"], "anchor": (-5.3692, 105.2433)}
    assert constraint_satisfaction_at_k(res, c, k=5) == 0.0


def test_empty_results():
    assert constraint_satisfaction_at_k([], {"gender": "putri"}, k=5) == 0.0
