"""FastAPI entry point untuk KozyNear search engine.

Architecture: single-container Docker. FastAPI serve:
- React SPA dari `/app/static/` (built dari frontend/)
- API endpoints di `/api/*`
- Health check di `/health` (Render uses this)
- Swagger UI di `/api/docs`

Routing order (FastAPI matches in registration order):
1. Specific API routes: /health, /api/info, /api/search, /api/listings/{id}, /api/eval/*
2. /api/docs, /api/openapi.json (FastAPI auto-generated)
3. /assets/* (Vite build output)
4. Catch-all /{path} -> serve index.html for SPA routing (React Router fallback)

Lifespan:
- Startup: load preprocessing pipeline + IR indexes dari disk
- Shutdown: cleanup
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.eval import router as eval_router
from app.api.health import router as health_router
from app.api.insights import router as insights_router
from app.api.listings import router as listings_router
from app.api.search import router as search_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load heavy resources. Shutdown: cleanup."""
    logger.info(f"[startup] KozyNear backend (env={settings.environment})")

    # Initialize app.state slots
    app.state.tfidf = None
    app.state.bm25 = None
    app.state.indobert = None
    app.state.hybrid = None
    app.state.preprocessing_pipeline = None
    app.state.gazetteer = None

    # 1. Preprocessing pipeline (Sastrawi factory init ~1s)
    try:
        from app.preprocessing import PreprocessingPipeline

        app.state.preprocessing_pipeline = PreprocessingPipeline()
        logger.info("[startup] preprocessing pipeline loaded")
    except Exception as e:
        logger.error(f"[startup] preprocessing pipeline FAIL: {e}")

    # 1b. Gazetteer (kampus + landmark) untuk smart search
    try:
        from app.search.gazetteer import Gazetteer

        app.state.gazetteer = Gazetteer.load()
        logger.info("[startup] gazetteer loaded")
    except Exception as e:
        app.state.gazetteer = None
        logger.error(f"[startup] gazetteer FAIL: {e}")

    # 1c. Listing cache in-memory: corpus statis (227 baris) di-load sekali,
    # menghapus SELECT lintas-region (HF di AS <-> Supabase di SG) dari jalur
    # panas /api/search?model=smart. DB down -> cache None, search fallback
    # query per-request.
    app.state.listings_cache = None
    try:
        from sqlalchemy import select as _select

        from app.core.db import async_session_factory
        from app.models.listing import Listing as _Listing

        async with async_session_factory() as _session:
            _rows = (await _session.execute(_select(_Listing))).scalars().all()
        if _rows:
            app.state.listings_cache = {r.id: r for r in _rows}
            logger.info(f"[startup] listing cache: {len(_rows)} rows in-memory")
        else:
            logger.warning("[startup] listing cache kosong (DB belum di-seed?)")
    except Exception as e:
        logger.warning(f"[startup] listing cache skip (DB unreachable): {e}")

    # 2. IR indexes (graceful skip kalau missing)
    try:
        from app.indexing.loader import load_all_indexes

        indexes_path = Path(settings.indexes_dir)
        if indexes_path.exists():
            indexes = load_all_indexes(
                indexes_path, include_neural=settings.enable_neural
            )
            app.state.tfidf = indexes.get("tfidf")
            app.state.bm25 = indexes.get("bm25")
            app.state.indobert = indexes.get("indobert")

            if app.state.bm25 is not None and app.state.indobert is not None:
                from app.indexing.hybrid import HybridIndex

                _pipeline = getattr(app.state, "preprocessing_pipeline", None)
                app.state.hybrid = HybridIndex(
                    app.state.bm25,
                    app.state.indobert,
                    query_preprocessor=(
                        (lambda q: _pipeline.process(q).processed)
                        if _pipeline is not None else None
                    ),
                )
                logger.info("[startup] hybrid index assembled")

            # BACKGROUND prewarm IndoBERT model:
            # - Index sudah loaded (embeddings.npy + faiss.index dari disk)
            # - SentenceTransformer model load butuh ~10-30s di Render free tier CPU
            # - SYNCHRONOUS prewarm di lifespan = block port binding (deploy fail)
            # - LAZY load di first request = Render proxy timeout 30s -> 502
            # - SOLUTION: spawn async background task setelah port binds
            # - Sambil model load di background, /search?model=indobert return
            #   503 dengan flag indobert_ready=False sampai task complete
            app.state.indobert_ready = False
            app.state.indobert_failed = False
            if app.state.indobert is not None:
                async def _bg_prewarm():
                    import asyncio
                    import time
                    try:
                        logger.info("[bg] starting IndoBERT model prewarm...")
                        t0 = time.perf_counter()
                        # Run blocking encode_query di thread pool supaya
                        # tidak block event loop
                        await asyncio.to_thread(
                            app.state.indobert.encode_query, "warmup",
                        )
                        elapsed = time.perf_counter() - t0
                        app.state.indobert_ready = True
                        logger.info(f"[bg] IndoBERT model warm ({elapsed:.1f}s)")
                    except Exception as e:
                        logger.error(f"[bg] IndoBERT prewarm FAIL: {e}")
                        # Set ke None supaya graceful 503 di /search
                        app.state.indobert = None
                        app.state.hybrid = None
                        app.state.indobert_failed = True

                import asyncio
                asyncio.create_task(_bg_prewarm())
                logger.info(
                    "[startup] IndoBERT prewarm scheduled di background; "
                    "/search?model=indobert akan return 503 sampai ready"
                )
        else:
            logger.warning(
                f"[startup] indexes_dir tidak ada: {indexes_path}. "
                "Build dulu via `python -m app.indexing.build`."
            )
    except Exception as e:
        logger.error(f"[startup] index loading FAIL: {e}")

    yield
    logger.info("[shutdown] KozyNear backend stopping")


app = FastAPI(
    title="KozyNear API",
    description=(
        "Indonesian Kos-Kosan Search Engine -- UNILA STKI Final Project. "
        "Compare 3 IR paradigms: TF-IDF, BM25, IndoBERT+FAISS, dan Hybrid."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS: tetap aktifkan untuk dev (frontend di :5173, backend di :8000).
# Di production single-container, frontend & API same origin -> CORS no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limit per-IP untuk endpoint komputasi (lihat app/core/ratelimit.py)
from app.core.ratelimit import rate_limit_middleware  # noqa: E402

app.middleware("http")(rate_limit_middleware)

# ----------------------------------------------------------------------------
# Routes order: API dulu, lalu static catch-all
# ----------------------------------------------------------------------------

# Health check at /health (Render uses this)
app.include_router(health_router, tags=["health"])

# API endpoints under /api prefix
app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(listings_router, prefix="/api", tags=["listings"])
app.include_router(eval_router, prefix="/api")  # eval_router internal prefix /eval
app.include_router(insights_router, prefix="/api", tags=["insights"])


# ----------------------------------------------------------------------------
# Static frontend (Docker build) -- serve dari /app/static
# ----------------------------------------------------------------------------
# Path discovery: di Docker container path absolute /app/static.
# Untuk local dev tanpa build, skip mount (frontend run terpisah via vite dev).
_DOCKER_STATIC = Path("/app/static")
_LOCAL_STATIC = Path(__file__).resolve().parent.parent.parent / "static"

STATIC_DIR: Path | None = None
if _DOCKER_STATIC.exists():
    STATIC_DIR = _DOCKER_STATIC
elif _LOCAL_STATIC.exists():
    STATIC_DIR = _LOCAL_STATIC

if STATIC_DIR is not None:
    # Mount /assets/ untuk JS/CSS chunks (Vite outputs ke /assets/)
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="assets",
        )

    # Catch-all: serve real file kalau ada, else index.html (SPA fallback)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Security: prevent path traversal
        candidate = (STATIC_DIR / full_path).resolve()
        if not str(candidate).startswith(str(STATIC_DIR.resolve())):
            raise HTTPException(403, "Forbidden")

        if candidate.is_file():
            return FileResponse(candidate)

        # SPA fallback: serve index.html untuk unknown routes
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(404, "Not Found")

    logger.info(f"[static] frontend mounted dari {STATIC_DIR}")
else:
    logger.info("[static] tidak ada static dir -- run frontend terpisah via vite dev")
