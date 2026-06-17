"""TF-IDF + cosine similarity index (scikit-learn).

Hyperparameter default:
- ngram_range=(1, 2): unigram + bigram (capture compound terms seperti
  "kamar mandi", "air panas")
- min_df=2: drop term yang muncul cuma sekali (likely typo/noise)
- max_features=10000: limit vocab size untuk memory + speed

Tim Anggota C: experiment dengan hyperparameter di
notebooks/03_model_comparison.ipynb. Pada corpus real 227 listing, TF-IDF
MAP ~0.24 (standard) / ~0.59 (pool-restricted) — kompetitif dengan BM25.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Document, IndexBase, SearchHit


class TFIDFIndex(IndexBase):
    name = "tfidf"

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 10000,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,  # log(1+tf) — handle high-tf terms
        )
        self.doc_matrix = None  # sparse matrix (N, vocab_size)
        self.doc_ids: list[str] = []
        self._size = 0

    def build(self, corpus: list[Document]) -> None:
        texts = [doc.text for doc in corpus]
        self.doc_ids = [doc.id for doc in corpus]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self._size = len(corpus)

    def query(self, q: str, top_k: int = 10) -> list[SearchHit]:
        if self.doc_matrix is None or self._size == 0:
            return []
        q_vec = self.vectorizer.transform([q])
        # Cosine similarity (since TF-IDF vectors already L2-normalized by default)
        scores = cosine_similarity(q_vec, self.doc_matrix).flatten()
        # Top-K indices (descending)
        top_k = min(top_k, self._size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        # Sort within top-K
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
                    "vectorizer": self.vectorizer,
                    "doc_matrix": self.doc_matrix,
                    "doc_ids": self.doc_ids,
                    "size": self._size,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TFIDFIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.vectorizer = data["vectorizer"]
        instance.doc_matrix = data["doc_matrix"]
        instance.doc_ids = data["doc_ids"]
        instance._size = data["size"]
        return instance
