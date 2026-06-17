"""Orkestrasi smart search: parse -> BM25 kandidat (+geo augment) -> fusion rank.

Dua layer:
- `smart_rank()` — core murni tanpa DB: terima dict listing yang sudah
  di-load. Dipakai endpoint (rows dari Postgres) DAN eval offline
  (rows dari JSONL), supaya yang dievaluasi = yang di-serve.
- `smart_search()` — wrapper async untuk endpoint: fetch listings dari DB
  lalu delegasi ke smart_rank. Corpus 227 baris, fetch-all per query masih
  murah (1 SELECT kecil) dan menghapus kelas bug "kandidat tidak ke-fetch".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.indexing.bm25 import BM25Index
from app.models.listing import Listing, ListingRead
from app.search.gazetteer import Gazetteer, haversine_km
from app.search.query_parser import parse
from app.search.ranker import Candidate, apply_hard_filter, fuse

OVERSHOOT = 5
# Bobot fusion (w_text, w_geo, w_attr) dari grid search simplex step 0.1
# (eval/smart_weights_grid.csv, n=30): 0.2/0.4/0.4 unggul atas default lama
# 0.4/0.4/0.2 di KEDUA lensa (CS@5 0.8633 vs 0.81, Wilcoxon p=0.023;
# MAP standard 0.3013 vs 0.2869). Kombinasi ber-CS lebih tinggi (mis.
# 0.1/0.2/0.7) ditolak: w_text terlalu kecil = overfit ke metric CS yang
# memang menghitung atribut; teks tetap perlu untuk query kualitas
# ("nyaman", "bersih") di luar pola constraint.
DEFAULT_WEIGHTS = (0.2, 0.4, 0.4)
# Saat query punya anchor ("dekat unila"), kandidat BM25 di-union dengan semua
# listing dalam radius ini. Tanpa augment, listing dekat anchor yang
# deskripsinya tidak menyebut nama anchor mustahil masuk hasil (skor geo cuma
# bisa menata ulang kandidat yang sudah ada, bukan menambah).
GEO_AUGMENT_RADIUS_KM = 3.0


@dataclass
class SmartFilters:
    """Filter eksplisit dari UI. Tidak pernah di-relax oleh hard filter."""
    harga_min: int | None = None
    harga_max: int | None = None
    tipe: str | None = None
    kecamatan: str | None = None


def _to_candidate(doc_id: str, text_score: float, row: Any) -> Candidate:
    return Candidate(
        doc_id=doc_id,
        text_score=text_score,
        tipe=row.tipe,
        harga=row.harga_per_bulan,
        fasilitas=row.fasilitas,
        lat=float(row.koordinat_lat) if row.koordinat_lat is not None else None,
        lng=float(row.koordinat_lng) if row.koordinat_lng is not None else None,
        kecamatan=row.kecamatan,
    )


def smart_rank(
    q: str,
    bm25: BM25Index,
    listings_by_id: Mapping[str, Any],
    gazetteer: Gazetteer,
    top_k: int = 10,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    preprocess: Callable[[str], str] | None = None,
    filters: SmartFilters | None = None,
) -> tuple[list[tuple[str, float]], dict, list[str]]:
    """Return (ranked [(doc_id, score)], understood, relaxed).

    listings_by_id: id -> objek dengan atribut tipe/harga_per_bulan/fasilitas/
    koordinat_lat/koordinat_lng/kecamatan (ORM row atau adapter eval).
    """
    parsed = parse(q, gazetteer)

    # Merge filter eksplisit: menang atas hasil parse, tandai di `explicit`.
    if filters is not None:
        if filters.tipe:
            parsed.gender = filters.tipe
            parsed.explicit.add("gender")
        if filters.harga_min is not None:
            parsed.harga_min = filters.harga_min
            parsed.explicit.add("harga_min")
        if filters.harga_max is not None:
            parsed.harga_max = filters.harga_max
            parsed.explicit.add("harga_max")
        if filters.kecamatan:
            parsed.kecamatan = filters.kecamatan
            parsed.explicit.add("kecamatan")
        parsed.build_understood()

    # Residual -> preprocessing yang SAMA dengan saat indexing (stem + jargon
    # + stopword), supaya token query match vocabulary index BM25.
    residual = parsed.residual_text or q
    text_q = preprocess(residual) if preprocess is not None else residual

    hits = bm25.query(text_q, top_k=top_k * OVERSHOOT) if text_q.strip() else []
    # Degenerate: residual kosong setelah preprocessing (mis. "kos murah")
    # atau BM25 tidak menemukan sinyal teks sama sekali -> semua listing jadi
    # kandidat, ranking diserahkan ke geo/atribut/harga.
    degenerate = not hits or all(h.score <= 0 for h in hits)

    score_map: dict[str, float] = {}
    if degenerate:
        score_map = {doc_id: 0.0 for doc_id in listings_by_id}
    else:
        score_map = {h.doc_id: h.score for h in hits}
        # Geo augment: union kandidat dengan listing radius N km dari anchor.
        if parsed.anchor is not None:
            for doc_id, row in listings_by_id.items():
                if doc_id in score_map:
                    continue
                if row.koordinat_lat is None or row.koordinat_lng is None:
                    continue
                dist = haversine_km(
                    float(row.koordinat_lat), float(row.koordinat_lng),
                    parsed.anchor.lat, parsed.anchor.lng,
                )
                if dist <= GEO_AUGMENT_RADIUS_KM:
                    score_map[doc_id] = 0.0

    cands: list[Candidate] = []
    for doc_id, text_score in score_map.items():
        row = listings_by_id.get(doc_id)
        if row is None:
            continue
        cands.append(_to_candidate(doc_id, text_score, row))

    kept, relaxed = apply_hard_filter(cands, parsed)
    ranked = fuse(kept, parsed, weights)

    # Tiebreak deterministik: skor sama -> harga termurah dulu. Penting untuk
    # degenerate path (semua skor 0) supaya hasil tidak arbitrer.
    harga_of = {c.doc_id: (c.harga if c.harga is not None else 10**12) for c in kept}
    ranked.sort(key=lambda t: (-t[1], harga_of.get(t[0], 10**12)))

    return ranked[:top_k], parsed.understood, relaxed


async def smart_search(
    q: str,
    bm25: BM25Index,
    session: AsyncSession,
    gazetteer: Gazetteer,
    top_k: int = 10,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    preprocess: Callable[[str], str] | None = None,
    filters: SmartFilters | None = None,
    listings_override: Mapping[str, Any] | None = None,
):
    """Return (results: list[ListingRead], understood: dict, relaxed: list[str]).

    listings_override: cache in-memory {id: Listing} dari app.state (diisi di
    lifespan). Kalau ada, SELECT per-request di-skip — menghapus roundtrip DB
    lintas-region (HF di AS, Supabase di SG) dari jalur panas pencarian.
    """
    if listings_override:
        by_id = listings_override
    else:
        rows = (await session.execute(select(Listing))).scalars().all()
        by_id = {r.id: r for r in rows}

    ranked, understood, relaxed = smart_rank(
        q, bm25, by_id, gazetteer,
        top_k=top_k, weights=weights, preprocess=preprocess, filters=filters,
    )

    results: list[ListingRead] = []
    for doc_id, score in ranked:
        r = by_id[doc_id]
        koord = (
            [float(r.koordinat_lat), float(r.koordinat_lng)]
            if r.koordinat_lat is not None and r.koordinat_lng is not None
            else None
        )
        results.append(ListingRead(
            id=r.id, judul=r.judul, deskripsi=r.deskripsi,
            harga_per_bulan=r.harga_per_bulan, tipe=r.tipe,
            fasilitas=r.fasilitas, alamat=r.alamat, kecamatan=r.kecamatan,
            score=score, koordinat=koord))
    return results, understood, relaxed
