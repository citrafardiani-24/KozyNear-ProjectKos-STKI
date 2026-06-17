"""Health check + status diagnostics endpoints.

- /health: simple liveness probe untuk Render
- /api/info: basic version info
- /api/status: comprehensive diagnostics (indexes, DB, model warm state)
  Penting untuk debug user-reported issues karena user copy-paste output ini.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe untuk Render.com."""
    return {"status": "ok", "service": "kozynear"}


@router.get("/api/info")
async def api_info():
    """Basic API info."""
    return {
        "name": "KozyNear API",
        "version": "0.2.0",
        "scope": "Bandar Lampung full coverage (20 kecamatan, 9 universitas)",
        "docs": "/api/docs",
        "health": "/health",
        "status": "/api/status",
    }


@router.get("/api/status")
async def status(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Comprehensive diagnostics — paste output ini saat report bug.

    Returns:
    - Service: name, version, environment, uptime
    - Indexes loaded (tfidf, bm25, indobert, hybrid)
    - Preprocessing pipeline loaded
    - Database: listings count, connection ok
    - Runtime: Python version, platform
    """
    from app.core.config import settings

    # Index availability checks
    state = request.app.state
    indexes_status = {
        "tfidf": state.tfidf is not None,
        "bm25": state.bm25 is not None,
        "indobert": state.indobert is not None,
        "hybrid": state.hybrid is not None,
    }
    indobert_ready = getattr(state, "indobert_ready", False)

    # Index sizes (kalau loaded)
    indexes_size = {}
    for name in ("tfidf", "bm25", "indobert"):
        idx = getattr(state, name, None)
        if idx is not None and hasattr(idx, "size"):
            try:
                indexes_size[name] = idx.size()
            except Exception:
                indexes_size[name] = None

    # DB row counts
    db_ok = True
    db_error = None
    listings_count = None
    try:
        from app.models.listing import Listing

        result = await session.execute(select(func.count(Listing.id)))
        listings_count = int(result.scalar() or 0)
    except Exception as e:
        db_ok = False
        db_error = str(e)

    # Memory usage (kalau psutil ada)
    memory_info = None
    try:
        import psutil

        process = psutil.Process()
        mem = process.memory_info()
        memory_info = {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "vms_mb": round(mem.vms / 1024 / 1024, 1),
        }
    except ImportError:
        pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": {
            "name": "kozynear",
            "version": "0.2.0",
            "environment": settings.environment,
        },
        "preprocessing": {
            "loaded": state.preprocessing_pipeline is not None,
        },
        "indexes": {
            "loaded": indexes_status,
            "sizes": indexes_size,
            "indexes_dir": settings.indexes_dir,
            "indobert_model_ready": indobert_ready,
        },
        "database": {
            "connected": db_ok,
            "listings_count": listings_count,
            "error": db_error,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "memory": memory_info,
        "settings": {
            "embedding_model": settings.embedding_model,
            "default_top_k": settings.default_top_k,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
        },
    }
