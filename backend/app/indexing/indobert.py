"""Neural / semantic index pakai sentence-transformers + FAISS.

Default model: `paraphrase-multilingual-MiniLM-L12-v2` (~118 MB, multilingual,
inferensi cepat di CPU). Untuk hasil lebih bagus tapi heavy: ganti ke
`indobenchmark/indobert-base-p2` (~440 MB).

FAISS IndexFlatIP = inner product. Karena embeddings sudah L2-normalized
(via `normalize_embeddings=True`), inner product == cosine similarity.

Tim Anggota D: experiment di notebook 03_model_comparison.ipynb:
- Compare MiniLM (small/fast) vs IndoBERT-base (better quality)
- Pooling strategy: mean-pool (default) vs CLS token
- FAISS Flat (exhaustive) vs IVF (clustered, faster tapi approximate)

Untuk corpus <= 5K docs, Flat sudah cukup (sub-ms query).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Document, IndexBase, SearchHit


class IndoBERTIndex(IndexBase):
    name = "indobert"

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.embeddings: Optional[np.ndarray] = None  # (N, dim) float32
        self.index = None  # faiss.IndexFlatIP
        self.doc_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        self._size = 0
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy load fastembed TextEmbedding (ONNX runtime).

        Trade-off vs sentence-transformers + torch:
        - fastembed: ~150MB total RAM (Render free tier compatible)
        - sentence-transformers + torch: ~500MB total (OOM di 512MB Render)
        - Same underlying MiniLM model, numerical results virtually identical
        - Faster init: ~1-2s vs 10-30s untuk first encode
        """
        if self._model is None:
            import os

            from fastembed import TextEmbedding

            # cache_dir eksplisit (FASTEMBED_CACHE_PATH) supaya model ONNX yang
            # di-pre-download saat docker build ke-reuse di runtime — tanpa ini
            # tiap cold start download ulang ~120MB.
            # `or None`: env kosong ("") harus berarti "pakai cache default
            # user", BUKAN cache_dir="" (= cwd; pernah bikin model terunduh
            # ke backend/ dan nyaris ter-commit).
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
            self._model = TextEmbedding(self.model_name, cache_dir=cache_dir)
        return self._model

    def _encode(self, texts: list[str]) -> "np.ndarray":
        """Encode texts via fastembed + L2-normalize untuk FAISS IndexFlatIP."""
        import numpy as np

        embeddings = list(self.model.embed(texts))
        arr = np.vstack(embeddings).astype("float32")
        if self.normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            arr = arr / norms
        return arr

    def build(self, corpus: list[Document]) -> None:
        import faiss

        texts = [(doc.raw_text or doc.text) for doc in corpus]
        embeddings = self._encode(texts)
        self.embeddings = embeddings
        # FAISS IndexFlatIP — exhaustive cosine via inner product
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.doc_ids = [doc.id for doc in corpus]
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self._size = len(corpus)

    def encode_query(self, q: str) -> np.ndarray:
        """Encode query string ke embedding (via fastembed ONNX).

        Returns shape (1, dim) — satu baris, bukan flat (dim,).
        FAISS index.search dan score_docs keduanya mengharapkan (1, dim).
        Jangan lakukan dot-product langsung tanpa .T transpose.
        """
        return self._encode([q])

    def score_docs(
        self, q_emb: np.ndarray, doc_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Score subset doc_ids vs query embedding using pre-computed embeddings.

        Dipakai oleh HybridIndex untuk rerank candidate dari BM25 tanpa
        re-encode dokumen.
        """
        if self.embeddings is None:
            return []
        indices = [self.id_to_idx[d] for d in doc_ids if d in self.id_to_idx]
        subset = self.embeddings[indices]
        # Inner product (cosine kalau normalized)
        scores = (subset @ q_emb.T).flatten()
        return [
            (self.doc_ids[idx], float(score))
            for idx, score in zip(indices, scores)
        ]

    def query(self, q: str, top_k: int = 10) -> list[SearchHit]:
        if self.index is None or self._size == 0:
            return []
        q_emb = self.encode_query(q)
        top_k = min(top_k, self._size)
        scores, indices = self.index.search(q_emb, top_k)
        return [
            SearchHit(
                doc_id=self.doc_ids[idx],
                score=float(scores[0][rank]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(indices[0])
            if idx >= 0  # FAISS pakai -1 untuk padding kalau result < top_k
        ]

    def save(self, path: Path) -> None:
        """Save: embeddings.npy + faiss.index + meta.json di folder `path`."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "normalize_embeddings": self.normalize_embeddings,
                    "batch_size": self.batch_size,
                    "doc_ids": self.doc_ids,
                    "size": self._size,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: Path) -> "IndoBERTIndex":
        """Load dari folder yang berisi embeddings.npy + faiss.index + meta.json."""
        import faiss

        with open(path / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        instance = cls(
            model_name=meta["model_name"],
            normalize_embeddings=meta["normalize_embeddings"],
            batch_size=meta["batch_size"],
        )
        instance.embeddings = np.load(path / "embeddings.npy")
        instance.index = faiss.read_index(str(path / "faiss.index"))
        instance.doc_ids = meta["doc_ids"]
        instance.id_to_idx = {d: i for i, d in enumerate(instance.doc_ids)}
        instance._size = meta["size"]
        return instance
