"""BM25 (Best Matching 25) index via rank_bm25.

Hyperparameter default sesuai standar literature:
- k1=1.5: term frequency saturation parameter
- b=0.75: length normalization parameter

Tim Anggota C: experiment k1 in [1.2, 2.0] dan b in [0.5, 1.0] untuk lihat
sensitivity. Biasanya BM25 beat TF-IDF tipis (~5-10% MAP) di Indonesian corpus.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .base import Document, IndexBase, SearchHit


class BM25Index(IndexBase):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._size = 0

    def build(self, corpus: list[Document]) -> None:
        self.doc_ids = [doc.id for doc in corpus]
        # rank_bm25 minta list of token lists
        self.tokenized_corpus = [doc.text.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self._size = len(corpus)

    def query(self, q: str, top_k: int = 10) -> list[SearchHit]:
        if self.bm25 is None or self._size == 0:
            return []
        q_tokens = q.split()
        scores = self.bm25.get_scores(q_tokens)
        top_k = min(top_k, self._size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [
            SearchHit(
                doc_id=self.doc_ids[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "doc_ids": self.doc_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "size": self._size,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(k1=data["k1"], b=data["b"])
        instance.doc_ids = data["doc_ids"]
        instance.tokenized_corpus = data["tokenized_corpus"]
        instance._size = data["size"]
        # Rebuild BM25Okapi (gak bisa di-pickle reliably across versions)
        instance.bm25 = BM25Okapi(
            instance.tokenized_corpus, k1=instance.k1, b=instance.b
        )
        return instance
