"""Hybrid retrieval: BM25 candidates -> IndoBERT rerank.

Strategy:
1. BM25 fetch top-N candidates (default N=50). BM25 cepat dan lexical-aware,
   bagus buat narrow down dari 1500-3000 docs ke 50.
2. IndoBERT score 50 candidates dengan pre-computed embeddings (no re-encode).
3. Combine score: `final = alpha * bm25_norm + (1 - alpha) * indobert_norm`,
   atau pure rerank by IndoBERT (alpha=0).
4. Return top-K (default K=10).

Tim Anggota D: experiment alpha in [0.0, 0.3, 0.5, 0.7] untuk lihat balance
yang optimal. Pure rerank (alpha=0) sering kompetitif untuk Indonesian
queries yang semantik.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .base import Document, IndexBase, SearchHit
from .bm25 import BM25Index
from .indobert import IndoBERTIndex


class HybridIndex(IndexBase):
    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Index,
        indobert: IndoBERTIndex,
        # None = rerank SEMUA dokumen (exhaustive). Plafon 50 yang lama
        # membuat neural mustahil menyelamatkan dokumen di luar top-50 BM25
        # (candidate recall ceiling); untuk corpus ratusan dokumen exhaustive
        # tetap murah karena embedding sudah precomputed.
        bm25_top_k: Optional[int] = None,
        # alpha 0.9 dari grid search pool-restricted n=30 (eval/
        # hybrid_alpha_grid_pool.csv): kurva flat 0.6-0.9, puncak 0.9
        # (selisih kecil, inconclusive). Default lama 0.3 menyeret hybrid
        # jatuh di standard eval karena GT lexical-pooled.
        alpha: float = 0.9,
        query_preprocessor: Optional[Callable[[str], str]] = None,
    ):
        """Args:
        bm25: BM25Index yang sudah di-build
        indobert: IndoBERTIndex yang sudah di-build
        bm25_top_k: jumlah candidate dari BM25 untuk re-rank
                    (None = semua dokumen, exhaustive)
        alpha: weight untuk BM25 score di final combination.
               alpha=0 -> pure IndoBERT rerank, alpha=1 -> pure BM25.
        query_preprocessor: callable(q_raw) -> q_processed; dipakai untuk BM25
            karena BM25 perlu teks ter-preprocess (stemmed). IndoBERT pakai
            raw query (natural language). Kalau None, q dipakai untuk
            keduanya (backwards-compat untuk pemanggil yang sudah preprocess).
        """
        self.bm25 = bm25
        self.indobert = indobert
        self.bm25_top_k = bm25_top_k
        self.alpha = alpha
        self.query_preprocessor = query_preprocessor
        self._size = bm25.size()

    def build(self, corpus: list[Document]) -> None:
        """Hybrid = composition; sub-indexes di-build separately."""
        # Tetap track size kalau dipanggil
        self._size = len(corpus)
        # NOTE: caller harus pastikan bm25 & indobert sudah di-build sebelum query.

    def query(self, q: str, top_k: int = 10) -> list[SearchHit]:
        # 1. BM25 candidates — pakai processed query (stemmed lexical match)
        q_for_bm25 = self.query_preprocessor(q) if self.query_preprocessor else q
        n_candidates = self.bm25_top_k or self.bm25.size()
        bm25_hits = self.bm25.query(q_for_bm25, top_k=n_candidates)
        if not bm25_hits:
            return []
        candidate_ids = [h.doc_id for h in bm25_hits]
        bm25_scores_raw = np.array([h.score for h in bm25_hits])

        # 2. IndoBERT score candidates pakai RAW query (semantic embedding)
        q_emb = self.indobert.encode_query(q)
        indobert_scores_pairs = self.indobert.score_docs(q_emb, candidate_ids)
        indobert_scores_map = dict(indobert_scores_pairs)

        # Align IndoBERT scores dengan order BM25 candidates
        indobert_scores_raw = np.array(
            [indobert_scores_map.get(doc_id, 0.0) for doc_id in candidate_ids]
        )

        # 3. Min-max normalize tiap score list (skala [0, 1])
        bm25_norm = self._minmax_normalize(bm25_scores_raw)
        indobert_norm = self._minmax_normalize(indobert_scores_raw)

        # 4. Combine
        combined = self.alpha * bm25_norm + (1 - self.alpha) * indobert_norm

        # 5. Sort + top-K
        sorted_idx = np.argsort(-combined)[:top_k]
        return [
            SearchHit(
                doc_id=candidate_ids[idx],
                score=float(combined[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(sorted_idx)
        ]

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize ke [0, 1]. Kalau semua sama, return zeros."""
        if len(scores) == 0:
            return scores
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-9:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def save(self, path: Path) -> None:
        """Hybrid bukan index murni — sub-indexes save separately."""
        raise NotImplementedError(
            "HybridIndex composition; save bm25 & indobert separately. "
            "Hybrid restore via constructor dari sub-indexes yang sudah di-load."
        )

    @classmethod
    def load(cls, path: Path) -> "HybridIndex":
        raise NotImplementedError(
            "HybridIndex composition; load bm25 & indobert separately, "
            "lalu instantiate HybridIndex(bm25, indobert)."
        )
