"""Unit test ranker: fusion geo/atribut + hard filter + fallback."""
from app.search.gazetteer import Anchor
from app.search.query_parser import ParsedQuery
from app.search.ranker import Candidate, apply_hard_filter, fuse

UNILA = Anchor("universitas lampung", -5.3692, 105.2433)


def _cand(doc_id, text, tipe, harga, fasilitas, lat, lng):
    return Candidate(doc_id, text, tipe, harga, fasilitas, lat, lng)


def test_geo_boost_orders_near_above_far():
    p = ParsedQuery(anchor=UNILA, residual_text="kos")
    near = _cand("a", 1.0, "putri", 800000, [], -5.3700, 105.2440)  # ~0.1km
    far = _cand("b", 1.0, "putri", 800000, [], -5.4100, 105.3200)   # jauh
    ranked = fuse([far, near], p, weights=(0.4, 0.4, 0.2))
    assert ranked[0][0] == "a"  # near menang walau text_score sama


def test_hard_filter_removes_wrong_gender():
    p = ParsedQuery(gender="putri")
    cands = [_cand("a", 1.0, "putra", 800000, [], None, None),
             _cand("b", 1.0, "putri", 800000, [], None, None)]
    kept, relaxed = apply_hard_filter(cands, p)
    assert [c.doc_id for c in kept] == ["b"]
    assert relaxed == []


def test_fallback_relaxes_when_empty():
    p = ParsedQuery(gender="putri", harga_max=500000)
    cands = [_cand("a", 1.0, "putri", 900000, [], None, None)]  # harga lewat
    kept, relaxed = apply_hard_filter(cands, p)
    assert len(kept) == 1  # dilonggarkan
    assert "harga" in relaxed
