"""Endpoint demo/insight: visualisasi preprocessing + statistik corpus.

Diporting dari konsep prototype Flask `proyek stki` (tab Preprocessing +
/api/stats) — fitur paling berguna untuk demo rubric Preprocessing & Sistem.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.models.listing import Listing

router = APIRouter()

# Fallback statistik kalau DB down: corpus.json ikut ter-copy di image.
_CORPUS_JSON = Path(__file__).resolve().parents[3] / "data" / "processed" / "corpus.json"


@router.get("/preprocess", tags=["insights"])
async def preprocess_trace(
    request: Request,
    text: str = Query(..., min_length=1, max_length=1000,
                      description="Teks untuk dilewatkan pipeline 9-stage"),
):
    """Jalankan pipeline preprocessing dengan trace per-stage (untuk UI demo)."""
    pipeline = getattr(request.app.state, "preprocessing_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, detail={"error": "preprocessing pipeline belum loaded"})
    result = pipeline.process(text, trace=True)
    return {
        "raw": result.raw,
        "processed": result.processed,
        "tokens": result.tokens,
        "extracted_prices": result.extracted_prices,
        "stages": result.trace,
    }


def _corpus_fallback_stats() -> dict:
    """Statistik dari corpus.json saat DB tidak tersedia (mis. test lokal)."""
    docs = json.loads(_CORPUS_JSON.read_text(encoding="utf-8"))
    kecamatan: dict[str, int] = {}
    tipe: dict[str, int] = {}
    hargas: list[int] = []
    for d in docs:
        meta = d.get("metadata", d)
        kec = meta.get("kecamatan") or "?"
        kecamatan[kec] = kecamatan.get(kec, 0) + 1
        t = meta.get("tipe") or "?"
        tipe[t] = tipe.get(t, 0) + 1
        h = meta.get("harga_per_bulan")
        if isinstance(h, int):
            hargas.append(h)
    return {
        "total_listings": len(docs),
        "kecamatan": dict(sorted(kecamatan.items(), key=lambda kv: -kv[1])),
        "tipe": tipe,
        "harga_min": min(hargas) if hargas else None,
        "harga_max": max(hargas) if hargas else None,
        "harga_avg": int(sum(hargas) / len(hargas)) if hargas else None,
        "source": "corpus.json (DB tidak tersedia)",
    }


@router.get("/stats", tags=["insights"])
async def corpus_stats(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Statistik corpus live: jumlah listing, distribusi kecamatan/tipe, harga,
    ukuran vocabulary index. Untuk tab Statistik + bahan slide."""
    stats: dict
    try:
        total = (await session.execute(select(func.count(Listing.id)))).scalar() or 0
        kec_rows = (await session.execute(
            select(Listing.kecamatan, func.count())
            .group_by(Listing.kecamatan)
            .order_by(func.count().desc())
        )).all()
        tipe_rows = (await session.execute(
            select(Listing.tipe, func.count()).group_by(Listing.tipe)
        )).all()
        harga_row = (await session.execute(
            select(
                func.min(Listing.harga_per_bulan),
                func.max(Listing.harga_per_bulan),
                func.avg(Listing.harga_per_bulan),
            )
        )).one()
        stats = {
            "total_listings": int(total),
            "kecamatan": {k or "?": int(c) for k, c in kec_rows},
            "tipe": {t or "?": int(c) for t, c in tipe_rows},
            "harga_min": int(harga_row[0]) if harga_row[0] is not None else None,
            "harga_max": int(harga_row[1]) if harga_row[1] is not None else None,
            "harga_avg": int(harga_row[2]) if harga_row[2] is not None else None,
            "source": "database",
        }
    except Exception as db_err:
        # DB down / belum seed -> fallback corpus committed (read-only stats)
        logger.warning(f"[stats] DB tidak tersedia, fallback corpus.json: {db_err}")
        try:
            stats = _corpus_fallback_stats()
        except Exception as e:  # corpus juga tidak ada
            logger.error(f"[stats] fallback corpus.json juga gagal: {e}")
            raise HTTPException(503, detail={"error": f"stats tidak tersedia: {e}"})

    # Vocabulary size dari index yang loaded (BM25 idf vocab)
    bm25 = getattr(request.app.state, "bm25", None)
    vocab_size = None
    if bm25 is not None and getattr(bm25, "bm25", None) is not None:
        try:
            vocab_size = len(bm25.bm25.idf)
        except Exception:
            vocab_size = None
    stats["vocab_size"] = vocab_size
    stats["models_loaded"] = [
        m for m in ("tfidf", "bm25", "indobert", "hybrid")
        if getattr(request.app.state, m, None) is not None
    ] + (["smart"] if getattr(request.app.state, "bm25", None) is not None
         and getattr(request.app.state, "gazetteer", None) is not None else [])
    return stats
