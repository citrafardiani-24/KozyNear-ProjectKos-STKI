"""Unit test query parser: ekstrak gender/harga/fasilitas/anchor dari query."""
from app.search.gazetteer import Gazetteer
from app.search.query_parser import parse

GZ = Gazetteer.load()


def test_parse_full_query():
    p = parse("kos cewe ac deket unila murah", GZ)
    assert p.gender == "putri"
    assert "ac" in p.fasilitas
    assert p.anchor is not None and p.anchor.name == "universitas lampung"
    assert p.harga_max == 1_000_000  # heuristik "murah"


def test_parse_explicit_price():
    p = parse("kos putra dekat itera maksimal 800rb", GZ)
    assert p.gender == "putra"
    assert p.harga_max == 800_000


def test_parse_gender_conflict_drops_gender():
    p = parse("kos putra putri campur", GZ)
    assert p.gender is None


def test_parse_plain_query_degrades():
    p = parse("kos bagus nyaman", GZ)
    assert p.gender is None and p.anchor is None and p.harga_max is None
    assert "kos" in p.residual_text
