"""Test rate limit middleware + listings_override di smart_search."""
import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config import settings
from app.indexing.base import Document
from app.indexing.bm25 import BM25Index
from app.search.gazetteer import Gazetteer
from app.search.pipeline import smart_search


def test_rate_limit_429_setelah_limit(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    # Reset state window antar test
    from app.core import ratelimit
    ratelimit._hits.clear()

    with TestClient(app) as client:
        codes = [
            client.get("/api/preprocess", params={"text": "kos"}).status_code
            for _ in range(4)
        ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_rate_limit_tidak_kena_endpoint_lain(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    from app.core import ratelimit
    ratelimit._hits.clear()

    with TestClient(app) as client:
        # /health bukan path terbatasi -> bebas berapa pun
        codes = [client.get("/health").status_code for _ in range(5)]
    assert codes == [200] * 5


class _ExplodingSession:
    """Session yang meledak kalau dipakai — bukti override dipakai."""

    async def execute(self, stmt):  # pragma: no cover - hanya guard
        raise AssertionError("DB tidak boleh disentuh saat ada listings_override")


def test_smart_search_pakai_override_tanpa_db():
    bm25 = BM25Index()
    bm25.build([Document(id="a", text="kos nyaman kampus"),
                Document(id="b", text="warung murah"),
                Document(id="c", text="warung bersih")])
    listing = SimpleNamespace(
        id="a", judul="Kos A", deskripsi="kos nyaman kampus",
        harga_per_bulan=700000, tipe="putri", fasilitas=["wifi"],
        alamat="Jl. X", kecamatan="Rajabasa",
        koordinat_lat=-5.37, koordinat_lng=105.244)
    results, understood, _ = asyncio.run(smart_search(
        "kos nyaman", bm25, _ExplodingSession(), Gazetteer.load(),
        top_k=3, listings_override={"a": listing}))
    assert [r.id for r in results] == ["a"]
