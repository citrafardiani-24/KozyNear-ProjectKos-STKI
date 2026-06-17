"""Unit test gazetteer: haversine + lookup anchor (kampus/landmark)."""
from app.search.gazetteer import Gazetteer, haversine_km


def test_haversine_known_distance():
    # UNILA ke ITERA sekitar 8 km (sanity: tidak meleset orde besaran)
    d = haversine_km(-5.3645, 105.2434, -5.3668, 105.3149)
    assert 5.0 < d < 12.0


def test_lookup_alias_unila():
    gz = Gazetteer.load()
    anchor = gz.lookup("kos murah deket unila")
    assert anchor is not None
    assert anchor.name == "universitas lampung"
    # Koordinat terverifikasi (wikipedia-id); toleransi kecil biar gak brittle
    assert abs(anchor.lat - (-5.3645)) < 0.01
    assert abs(anchor.lng - 105.2434) < 0.01


def test_anchors_within_bandar_lampung_bbox():
    """Guard anti-placeholder: semua anchor wajib dalam bbox Bandar Lampung raya."""
    gz = Gazetteer.load()
    anchors = {a for _, a in gz._pairs}
    assert len(anchors) >= 10
    for a in anchors:
        assert -5.55 <= a.lat <= -5.20, f"{a.name}: lat {a.lat} di luar bbox"
        assert 105.10 <= a.lng <= 105.40, f"{a.name}: lng {a.lng} di luar bbox"


def test_lookup_new_verified_anchors():
    gz = Gazetteer.load()
    teknokrat = gz.lookup("kos dekat teknokrat")
    malahayati = gz.lookup("kos deket unmal")
    assert teknokrat is not None and abs(teknokrat.lat - (-5.3824)) < 0.01
    # Malahayati di Kemiling (barat kota) — dulu placeholder salah 9 km
    assert malahayati is not None and abs(malahayati.lng - 105.2187) < 0.01


def test_lookup_landmark_mbk():
    gz = Gazetteer.load()
    anchor = gz.lookup("kos deket mbk")
    assert anchor is not None
    assert "boemi kedaton" in anchor.name


def test_lookup_none_when_no_anchor():
    gz = Gazetteer.load()
    assert gz.lookup("kos murah ac") is None
