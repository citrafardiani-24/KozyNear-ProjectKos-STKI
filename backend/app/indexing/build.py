"""CLI builder untuk semua IR indexes.

Workflow:
    # 1. Setelah scraping selesai dan preprocessing dijalankan:
    #    data/processed/corpus.json berisi list of {id, text, raw_text, metadata}

    # 2. Build semua index sekaligus
    python -m app.indexing.build \\
        --corpus ../data/processed/corpus.json \\
        --output-dir ../data/indexes

    # 3. Output struktur:
    #    data/indexes/tfidf.pkl
    #    data/indexes/bm25.pkl
    #    data/indexes/indobert/embeddings.npy
    #    data/indexes/indobert/faiss.index
    #    data/indexes/indobert/meta.json

Tiap index loadable independen di FastAPI lifespan startup (lihat loader.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from .base import Document
from .bm25 import BM25Index
from .indobert import IndoBERTIndex
from .tfidf import TFIDFIndex


def load_corpus(path: Path) -> list[Document]:
    """Load corpus JSON: expect list of {id, text, raw_text?, metadata?}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Document(
            id=item["id"],
            text=item["text"],
            raw_text=item.get("raw_text"),
            metadata=item.get("metadata", {}),
        )
        for item in raw
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IR indexes dari corpus")
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="Path ke corpus.json (output preprocessing)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Folder output indexes (e.g., ../data/indexes)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["tfidf", "bm25", "indobert"],
        help="Index yang di-skip (untuk debug)",
    )
    parser.add_argument(
        "--indobert-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name (default: MiniLM lightweight)",
    )
    parser.add_argument(
        "--tfidf-ngram", nargs=2, type=int, default=[1, 2],
        help="TF-IDF ngram range (default: 1 2)",
    )
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    args = parser.parse_args()

    logger.info(f"[load corpus] {args.corpus}")
    corpus = load_corpus(args.corpus)
    logger.info(f"[corpus] {len(corpus)} documents")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- TF-IDF ----
    if "tfidf" not in args.skip:
        logger.info("[build TF-IDF]")
        t0 = time.time()
        tfidf = TFIDFIndex(ngram_range=tuple(args.tfidf_ngram))
        tfidf.build(corpus)
        tfidf.save(args.output_dir / "tfidf.pkl")
        logger.info(f"[TF-IDF] saved ({time.time() - t0:.1f}s)")

    # ---- BM25 ----
    if "bm25" not in args.skip:
        logger.info("[build BM25]")
        t0 = time.time()
        bm25 = BM25Index(k1=args.bm25_k1, b=args.bm25_b)
        bm25.build(corpus)
        bm25.save(args.output_dir / "bm25.pkl")
        logger.info(f"[BM25] saved ({time.time() - t0:.1f}s)")

    # ---- IndoBERT + FAISS ----
    if "indobert" not in args.skip:
        logger.info(f"[build IndoBERT] model={args.indobert_model}")
        t0 = time.time()
        indobert = IndoBERTIndex(model_name=args.indobert_model)
        indobert.build(corpus)
        indobert.save(args.output_dir / "indobert")
        logger.info(f"[IndoBERT] saved ({time.time() - t0:.1f}s)")

    logger.info(f"[done] all indexes saved ke {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
