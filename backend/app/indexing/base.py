"""Abstract IndexBase + shared dataclasses.

Setiap index subclass implement: build, query, save, load.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Document:
    """Document representation untuk indexing.

    `text` = processed text (output preprocessing pipeline), siap di-index.
    `raw_text` = original deskripsi (untuk display di UI).
    `metadata` = field tambahan (judul, harga, alamat, kecamatan, dll).
    """
    id: str
    text: str
    raw_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """Single result dari index query."""
    doc_id: str
    score: float
    rank: int


class IndexBase(ABC):
    """Abstract base class untuk semua IR index.

    Convention: setiap subclass punya class attribute `name` (string identifier
    untuk routing di /search?model=<name>).
    """

    name: str = "base"  # override per subclass

    @abstractmethod
    def build(self, corpus: list[Document]) -> None:
        """Build index dari corpus. Overwrite existing state."""

    @abstractmethod
    def query(self, q: str, top_k: int = 10) -> list[SearchHit]:
        """Run query, return top-K hits sorted by score DESC."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize index ke disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "IndexBase":
        """Deserialize index dari disk."""

    def size(self) -> int:
        """Jumlah dokumen yang sudah di-index."""
        return getattr(self, "_size", 0)
