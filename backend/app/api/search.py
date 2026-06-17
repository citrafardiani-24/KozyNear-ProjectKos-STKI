"""Search endpoint: GET /search?q=X&model=Y&top_k=N + optional filters.

Pipeline:
1. Pilih IR index dari app.state berdasar param `model`
2. Preprocess query (same as corpus preprocessing saat indexing)
3. IR index query top-K * overshoot (3x) — supaya filter setelahnya
   gak return < top_k
4. Hydrate dari DB dengan filter (harga, tipe, kecamatan)
5. Take top-K filtered, preserve ranking, return SearchResponse
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.models.listing import Listing, ListingRead
from app.models.search import SearchResponse
from app.search.pipeline import SmartFilters, smart_search


router = APIRouter()


# Overshoot factor: index query ambil top_k * X supaya filter gak terlalu
# memotong hasil. 3 cukup untuk filter ringan; bisa naik kalau filter ketat.
OVERSHOOT_FACTOR = 3


@router.get("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Query string"),
    model: Literal["tfidf", "bm25", "indobert", "hybrid", "smart"] = Query(
        "smart", description="IR model pilihan (default: smart = query understanding + geo + BM25)"
    ),
    top_k: int = Query(10, ge=1, le=50, description="Jumlah hasil"),
    # ---- Optional filters (metadata-based) ----
    harga_min: Optional[int] = Query(
        None, ge=0, description="Min harga per bulan (IDR)"
    ),
    harga_max: Optional[int] = Query(
        None, ge=0, description="Max harga per bulan (IDR)"
    ),
    tipe: Optional[Literal["putra", "putri", "campur"]] = Query(
        None, description="Filter tipe kos"
    ),
    kecamatan: Optional[str] = Query(
        None, description="Filter kecamatan (substring match)"
    ),
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    """Cari kos dengan IR model + optional metadata filters.

    Returns 503 kalau index belum di-load di backend.
    """
    # 0. Smart pipeline (default): query understanding + geo + BM25 (no neural).
    if model == "smart":
        gz = getattr(request.app.state, "gazetteer", None)
        bm25_index = getattr(request.app.state, "bm25", None)
        if bm25_index is None or gz is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "smart pipeline belum siap (butuh bm25 + gazetteer)"},
            )
        # Residual query harus lewat preprocessing yang sama dengan corpus
        # saat indexing (stem + jargon), kalau tidak token gak match index.
        pipeline = getattr(request.app.state, "preprocessing_pipeline", None)
        preprocess = (
            (lambda s: pipeline.process(s).processed) if pipeline else None
        )
        # Filter UI eksplisit ikut ke smart sebagai hard constraint
        # (sebelumnya diabaikan diam-diam di mode smart).
        smart_filters = SmartFilters(
            harga_min=harga_min, harga_max=harga_max,
            tipe=tipe, kecamatan=kecamatan,
        )
        t0 = time.perf_counter()
        results, understood, relaxed = await smart_search(
            q, bm25_index, session, gz, top_k=top_k,
            preprocess=preprocess, filters=smart_filters,
            listings_override=getattr(request.app.state, "listings_cache", None),
        )
        return SearchResponse(
            query=q, model=model, top_k=top_k,
            took_ms=int((time.perf_counter() - t0) * 1000),
            results=results, understood=understood, relaxed=relaxed,
        )

    # 1. Get index dari app.state
    index = getattr(request.app.state, model, None)
    if index is None:
        available = [
            m for m in ("tfidf", "bm25", "indobert", "hybrid")
            if getattr(request.app.state, m, None) is not None
        ]
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Index '{model}' belum di-load di backend",
                "model_requested": model,
                "available_models": available,
                "preprocessing_loaded": getattr(
                    request.app.state, "preprocessing_pipeline", None
                ) is not None,
                "hint": (
                    "Kemungkinan OOM saat prewarm IndoBERT di Render free tier "
                    "(512MB RAM). Check Render logs untuk detail. Untuk "
                    "diagnostics lengkap GET /api/status."
                ),
            },
        )

    # 1b. Check IndoBERT/Hybrid model warm state (background prewarm)
    if model in ("indobert", "hybrid"):
        if not getattr(request.app.state, "indobert_ready", False):
            failed = getattr(request.app.state, "indobert_failed", False)
            if failed:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "IndoBERT model gagal load (prewarm permanent failure)",
                        "model_requested": model,
                        "ready_models": [
                            m for m in ("tfidf", "bm25")
                            if getattr(request.app.state, m, None) is not None
                        ],
                        "hint": (
                            "IndoBERT prewarm gagal permanen (kemungkinan OOM di "
                            "Render free tier 512MB). Restart deployment atau gunakan "
                            "TF-IDF/BM25. Retry TIDAK akan membantu tanpa restart."
                        ),
                    },
                )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "IndoBERT model masih loading (background prewarm)",
                    "model_requested": model,
                    "ready_models": [
                        m for m in ("tfidf", "bm25")
                        if getattr(request.app.state, m, None) is not None
                    ],
                    "hint": (
                        "Background prewarm jalan di startup, butuh ~10-30 detik "
                        "di Render free tier. Coba TF-IDF/BM25 dulu yang sudah "
                        "ready, atau retry IndoBERT/Hybrid setelah ~30 detik."
                    ),
                    "retry_after_sec": 30,
                },
                headers={"Retry-After": "30"},
            )

    t0 = time.perf_counter()

    # 2. Preprocess query (untuk BM25/TF-IDF). IndoBERT & Hybrid pakai raw
    # query karena sentence-transformers ekspektasi natural language.
    pipeline = getattr(request.app.state, "preprocessing_pipeline", None)
    processed_q = pipeline.process(q).processed if pipeline else q
    search_q = q if model in ("indobert", "hybrid") else processed_q
    logger.debug(
        f"[search] q='{q}' processed='{processed_q}' model={model} "
        f"using={'raw' if model in ('indobert', 'hybrid') else 'processed'}"
    )

    # 3. IR query — overshoot kalau ada filter (mitigate filter cutting hits)
    has_filter = any([harga_min is not None, harga_max is not None, tipe is not None, kecamatan is not None])
    fetch_k = top_k * OVERSHOOT_FACTOR if has_filter else top_k
    hits = index.query(search_q, top_k=fetch_k)
    if not hits:
        return SearchResponse(
            query=q, model=model, top_k=top_k, took_ms=0, results=[]
        )

    doc_ids = [h.doc_id for h in hits]
    score_map = {h.doc_id: h.score for h in hits}

    # 4. Hydrate from DB dengan filters
    stmt = select(Listing).where(Listing.id.in_(doc_ids))
    if harga_min is not None:
        stmt = stmt.where(Listing.harga_per_bulan >= harga_min)
    if harga_max is not None:
        stmt = stmt.where(Listing.harga_per_bulan <= harga_max)
    if tipe:
        stmt = stmt.where(Listing.tipe == tipe)
    if kecamatan:
        stmt = stmt.where(Listing.kecamatan.ilike(f"%{kecamatan}%"))

    db_result = await session.execute(stmt)
    listings_map = {row.id: row for row in db_result.scalars().all()}

    # 5. Preserve IR ranking, take top_k filtered
    results: list[ListingRead] = []
    for doc_id in doc_ids:
        if len(results) >= top_k:
            break
        listing = listings_map.get(doc_id)
        if not listing:
            continue  # filtered out atau gak ada di DB
        results.append(
            ListingRead(
                id=listing.id,
                judul=listing.judul,
                deskripsi=listing.deskripsi,
                harga_per_bulan=listing.harga_per_bulan,
                tipe=listing.tipe,
                fasilitas=listing.fasilitas,
                alamat=listing.alamat,
                kecamatan=listing.kecamatan,
                score=score_map[doc_id],
                koordinat=(
                    [float(listing.koordinat_lat), float(listing.koordinat_lng)]
                    if listing.koordinat_lat is not None
                    and listing.koordinat_lng is not None
                    else None
                ),
            )
        )

    took_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        f"[search] q='{q}' model={model} filter={has_filter} "
        f"hits={len(results)}/{top_k} took_ms={took_ms}"
    )
    return SearchResponse(
        query=q, model=model, top_k=top_k, took_ms=took_ms, results=results
    )
