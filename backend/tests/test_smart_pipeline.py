"""Test smart pipeline: smart_rank core (DB-free) + smart_search wrapper."""
import asyncio
from types import SimpleNamespace

from app.indexing.base import Document
from app.indexing.bm25 import BM25Index
from app.search.gazetteer import Gazetteer
from app.search.pipeline import SmartFilters, smart_rank, smart_search


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, stmt):
        return _FakeResult(self._rows)


def _listing(doc_id, tipe, lat, lng, harga=800000, kecamatan="rajabasa",
             fasilitas=None):
    return SimpleNamespace(
        id=doc_id, judul=f"Kos {doc_id}", deskripsi="kos nyaman dekat kampus",
        harga_per_bulan=harga, tipe=tipe,
        fasilitas=fasilitas if fasilitas is not None else ["ac", "wifi"],
        alamat="Jl. Contoh", kecamatan=kecamatan,
        koordinat_lat=lat, koordinat_lng=lng)


def _bm25(docs):
    idx = BM25Index()
    idx.build(docs)
    return idx


GZ = Gazetteer.load()


def test_smart_search_ranks_near_first():
    bm25 = _bm25([Document(id="a", text="kos nyaman dekat kampus"),
                  Document(id="b", text="kos nyaman dekat kampus")])
    rows = [_listing("a", "putri", -5.3700, 105.2440),   # dekat unila
            _listing("b", "putri", -5.4100, 105.3200)]    # jauh
    results, understood, relaxed = asyncio.run(smart_search(
        "kos putri deket unila", bm25, _FakeSession(rows), GZ, top_k=2))
    assert results[0].id == "a"
    assert understood["anchor"] == "universitas lampung"


def test_explicit_tipe_filter_overrides_parsed_gender():
    """Filter UI tipe=putra menang atas kata 'putri' di query, tanpa relax."""
    bm25 = _bm25([Document(id="a", text="kos nyaman kampus"),
                  Document(id="b", text="kos nyaman kampus")])
    by_id = {
        "a": _listing("a", "putri", -5.37, 105.244),
        "b": _listing("b", "putra", -5.37, 105.244),
    }
    ranked, understood, relaxed = smart_rank(
        "kos putri nyaman", bm25, by_id, GZ,
        filters=SmartFilters(tipe="putra"))
    ids = [doc_id for doc_id, _ in ranked]
    assert ids == ["b"]
    assert understood["gender"] == "putra"
    assert "gender" not in relaxed  # eksplisit tidak pernah di-relax


def test_explicit_harga_min_and_kecamatan_enforced():
    bm25 = _bm25([Document(id=i, text="kos nyaman kampus") for i in "abc"])
    by_id = {
        "a": _listing("a", "campur", -5.37, 105.244, harga=400000, kecamatan="Kedaton"),
        "b": _listing("b", "campur", -5.37, 105.244, harga=900000, kecamatan="Rajabasa"),
        "c": _listing("c", "campur", -5.37, 105.244, harga=950000, kecamatan="Kedaton"),
    }
    ranked, _, _ = smart_rank(
        "kos nyaman", bm25, by_id, GZ,
        filters=SmartFilters(harga_min=500000, kecamatan="kedaton"))
    ids = [doc_id for doc_id, _ in ranked]
    assert ids == ["c"]  # a gugur harga_min, b gugur kecamatan


def test_degenerate_query_falls_back_to_cheapest():
    """'kos murah' residualnya kosong -> jangan arbitrer, urut harga naik."""
    bm25 = _bm25([Document(id="a", text="nyaman kampus unila"),
                  Document(id="b", text="bersih strategis")])
    by_id = {
        "a": _listing("a", "campur", -5.37, 105.244, harga=700000),
        "b": _listing("b", "campur", -5.37, 105.244, harga=500000),
    }
    # preprocess yang membuang semua token (simulasi 'kos murah' -> '')
    ranked, understood, _ = smart_rank(
        "kos murah", bm25, by_id, GZ, preprocess=lambda s: "")
    ids = [doc_id for doc_id, _ in ranked]
    assert ids == ["b", "a"]  # termurah dulu, dua-duanya <= 1 juta (heuristik murah)
    assert understood["harga_max"] == 1_000_000


def test_geo_augment_includes_unmentioned_listing_near_anchor():
    """Listing dekat anchor yang TIDAK menyebut nama anchor tetap masuk."""
    bm25 = _bm25([Document(id="text-match", text="kos dekat unila kampus"),
                  Document(id="silent-near", text="hunian bersih tenang")])
    by_id = {
        "text-match": _listing("text-match", "campur", -5.4100, 105.3200),  # jauh
        "silent-near": _listing("silent-near", "campur", -5.3660, 105.2450),  # ~250m dari unila
    }
    ranked, _, _ = smart_rank("kos deket unila", bm25, by_id, GZ, top_k=5)
    ids = [doc_id for doc_id, _ in ranked]
    assert "silent-near" in ids  # tanpa augment dia mustahil jadi kandidat


def test_preprocess_applied_to_residual():
    """Residual harus lewat preprocessor sebelum BM25 (match vocab index)."""
    # 3 dokumen: dengan 2 dokumen, df=1 dari N=2 bikin IDF Okapi = ln(1) = 0
    # dan skor jadi 0 (ketabrak deteksi degenerate).
    bm25 = _bm25([Document(id="a", text="kampus nyaman"),
                  Document(id="b", text="warung murah"),
                  Document(id="c", text="warung bersih")])
    by_id = {
        "a": _listing("a", "campur", None, None),
        "b": _listing("b", "campur", None, None),
        "c": _listing("c", "campur", None, None),
    }
    # tanpa preprocess: token 'kampusnya' tidak ada di index -> degenerate
    fake_stem = lambda s: s.replace("kampusnya", "kampus")  # noqa: E731
    ranked, _, _ = smart_rank("kampusnya", bm25, by_id, GZ, preprocess=fake_stem)
    assert ranked[0][0] == "a"
    assert ranked[0][1] > 0  # ada sinyal teks, bukan jalur degenerate


def test_relaxed_reports_only_parsed_constraints():
    """Constraint parse yang bikin kosong dilonggarkan + dilaporkan."""
    bm25 = _bm25([Document(id="a", text="kos nyaman kampus")])
    by_id = {"a": _listing("a", "putra", -5.37, 105.244, harga=2_000_000)}
    # parsed: gender=putri (mismatch) + murah (harga > 1jt) -> dua-duanya relax
    ranked, _, relaxed = smart_rank("kos putri murah nyaman", bm25, by_id, GZ)
    assert [doc_id for doc_id, _ in ranked] == ["a"]
    assert set(relaxed) == {"harga", "gender"}
