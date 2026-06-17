"""Loader helper untuk FastAPI lifespan startup.

Load semua indexes dari disk sekali saat backend startup, store di
`app.state` untuk reuse di route handlers.

Usage di app/main.py:

    from app.indexing.loader import load_all_indexes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        indexes = load_all_indexes(Path(settings.indexes_dir))
        app.state.tfidf = indexes["tfidf"]
        app.state.bm25 = indexes["bm25"]
        app.state.indobert = indexes["indobert"]
        app.state.hybrid = HybridIndex(indexes["bm25"], indexes["indobert"])
        yield
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from .bm25 import BM25Index
from .tfidf import TFIDFIndex


def load_all_indexes(
    indexes_dir: Path, include_neural: bool = True
) -> dict[str, object]:
    """Load semua indexes dari folder. Return dict {name: index_instance}.

    Missing index logged sebagai warning, tidak raise (supaya partial deploy OK).
    `include_neural=False` melewati IndoBERT supaya runtime production tidak
    perlu import faiss/fastembed (hemat RAM).
    """
    result: dict[str, object] = {}

    tfidf_path = indexes_dir / "tfidf.pkl"
    if tfidf_path.exists():
        logger.info(f"[load] TF-IDF dari {tfidf_path}")
        result["tfidf"] = TFIDFIndex.load(tfidf_path)
    else:
        logger.warning(f"[skip] TF-IDF tidak ditemukan di {tfidf_path}")

    bm25_path = indexes_dir / "bm25.pkl"
    if bm25_path.exists():
        logger.info(f"[load] BM25 dari {bm25_path}")
        result["bm25"] = BM25Index.load(bm25_path)
    else:
        logger.warning(f"[skip] BM25 tidak ditemukan di {bm25_path}")

    indobert_path = indexes_dir / "indobert"
    if not include_neural:
        logger.info("[skip] IndoBERT dilewati (enable_neural=False)")
    elif indobert_path.exists() and indobert_path.is_dir():
        from .indobert import IndoBERTIndex  # lazy: hindari import faiss saat off

        logger.info(f"[load] IndoBERT dari {indobert_path}")
        result["indobert"] = IndoBERTIndex.load(indobert_path)
    else:
        logger.warning(f"[skip] IndoBERT tidak ditemukan di {indobert_path}")

    logger.info(f"[load] {len(result)} indexes loaded: {list(result.keys())}")
    return result
