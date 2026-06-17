"""IR index builders: TF-IDF, BM25, IndoBERT+FAISS, Hybrid.

Public API:
    from app.indexing import (
        Document, SearchHit, IndexBase,
        TFIDFIndex, BM25Index, IndoBERTIndex, HybridIndex,
    )

Workflow:
    1. Load corpus dari data/processed/corpus.json (after preprocessing)
    2. Build setiap index, save ke data/indexes/
    3. FastAPI lifespan load semua indexes saat startup
    4. Search route /search?model=X panggil index.query(q)
"""

from .base import Document, IndexBase, SearchHit
from .bm25 import BM25Index
from .hybrid import HybridIndex
from .indobert import IndoBERTIndex
from .tfidf import TFIDFIndex

__all__ = [
    "Document",
    "SearchHit",
    "IndexBase",
    "TFIDFIndex",
    "BM25Index",
    "IndoBERTIndex",
    "HybridIndex",
]
